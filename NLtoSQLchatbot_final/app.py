"""
Flask와 LangGraph를 이용한 AI 챗봇 애플리케이션

이 애플리케이션은 다음 기능을 제공합니다:
1.  사용자 질문의 의도를 파악하고, 관련 DB 스키마와 가이드를 검색합니다.
2.  검색된 정보를 바탕으로 LLM(gpt-4o)을 사용하여 SQL 쿼리를 생성합니다.
3.  생성된 SQL을 Oracle DB에서 실행하고, 결과를 사용자에게 요약하여 제공합니다.
4.  LangGraph를 사용하여 위 과정을 체계적인 워크플로우로 구성합니다.
5.  Flask를 통해 웹 인터페이스와 사용자 관리(로그인, 회원가입) API를 제공합니다.
"""
import os
import re
import json
import bcrypt
import cx_Oracle
from typing import TypedDict, List, Dict, Any

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END

# --- 1. 설정: 환경 변수 및 기본 설정 ---

# Langsmith 및 OpenAI API 키 설정 (디버깅 및 추적용)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"] = "본인의 api-key"
os.environ["LANGSMITH_PROJECT"] = "프로젝트 이름"
os.environ["OPENAI_API_KEY"] = "본인의 api-key"

# Flask 애플리케이션 초기화 및 CORS 설정
app = Flask(__name__)
CORS(app)

# Oracle DB 연결 정보 (환경 변수에서 가져오기)
ORACLE_USER = os.getenv("ORACLE_USER", "유저이름")
ORACLE_PW = os.getenv("ORACLE_PW", "패스워드")
ORACLE_HOST = os.getenv("ORACLE_HOST", "호스트이름")
ORACLE_PORT = int(os.getenv("ORACLE_PORT", "포트번호_보통1521"))
ORACLE_SERVICE = os.getenv("ORACLE_SERVICE", "서비스이름")

# --- 2. LangChain 및 기타 전역 객체 초기화 ---

# 언어 모델(LLM), 임베딩 모델, 벡터DB 전역 초기화
llm = ChatOpenAI(model="gpt-4o", temperature=0)
embedding = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
vectordb = Chroma(persist_directory="./chroma_db", embedding_function=embedding)

# 의도별로 사용할 스키마 파일과 가이드라인을 매핑한 JSON 파일 로드
with open('guide_map.json', encoding='utf-8') as f:
    GUIDE_MAP = json.load(f)


# --- 3. 핵심 로직 헬퍼 함수 정의 ---
# 이 함수들은 LangGraph의 노드 안에서 재사용됩니다.

def infer_intent(question: str) -> str:
    """
    사용자 질문에 포함된 키워드를 기반으로 의도를 추론합니다.

    Args:
        question (str): 사용자 질문 문자열.

    Returns:
        str: 추론된 의도 (예: '임상시험/AE', '바이탈/검사').
    """
    q = question.lower()
    if any(w in q for w in ['임상시험', 'inclusion', 'exclusion', '제외', 'ae', 'adr', 'susar']):
        if 'ae' in q or 'adr' in q or 'susar' in q: return '임상시험/AE'
        if '제외' in q or 'exclusion' in q: return '임상시험/제외'
        return '임상시험'
    if any(w in q for w in ['혈압', '맥박', '체온', '혈당', 'wbc', 'hb', 'glucose', 'chart', 'lab', '검사']): return '바이탈/검사'
    if any(w in q for w in ['진단', 'icd', '코드', '시술', 'procedure', '수술']): return '진단/시술'
    if any(w in q for w in ['약', '투약', 'drug', '처방', 'medication', '항생제']): return '약물/투약'
    if any(w in q for w in ['수액', '투여', 'infusion', 'fluid']): return '수액/투여'
    if any(w in q for w in ['미생물', '감염', '균', 'infection']): return '미생물/감염'
    if any(w in q for w in ['icu', '중환자', '재원', 'los']): return 'ICU/재원'
    if any(w in q for w in ['입원', 'admit', '퇴원', 'discharge']): return '환자/입원'
    return '기본'


def load_schema_and_guide(intent: str) -> (str, str):
    """
    추론된 의도에 따라 필요한 DB 스키마와 가이드라인을 로드합니다.

    Args:
        intent (str): infer_intent 함수로 추론된 의도.

    Returns:
        tuple: (스키마 컨텍스트, 가이드라인 텍스트).
    """
    guide_item = GUIDE_MAP.get(intent, GUIDE_MAP.get('기본', {}))
    schema_files = guide_item.get("schema", ["schema_patients.txt"])
    guide = guide_item.get("guide", "")
    context = ""
    for fname in schema_files:
        try:
            with open(fname, encoding='utf-8') as f:
                context += f"\n[{fname}]\n" + f.read() + "\n"
        except FileNotFoundError:
            continue
    return context, guide


def extract_faq_from_context(context: str) -> List[str]:
    """
    주어진 텍스트 컨텍스트에서 Q/A 형식의 FAQ 목록을 정규식으로 추출합니다.

    Args:
        context (str): FAQ가 포함된 텍스트.

    Returns:
        list: 추출된 "Q: ... A: ..." 형식의 문자열 리스트.
    """
    faq_list = []
    faq_pairs = re.findall(r"Q[:：](.*?)\nA[:：](.*?)(?=\nQ[:：]|\Z)", context, re.DOTALL)
    for q, a in faq_pairs:
        faq_list.append("Q:" + q.strip() + "\nA:" + a.strip())
    return faq_list


def hybrid_search(query: str, vectordb: Chroma, keyword_corpus: List[str], top_k: int = 3) -> List[Any]:
    """
    벡터 검색과 키워드 검색을 결합한 하이브리드 검색을 수행합니다.

    Args:
        query (str): 사용자 검색어.
        vectordb (Chroma): 검색을 수행할 Chroma 벡터DB 객체.
        keyword_corpus (list): 키워드 검색을 위한 FAQ 등 텍스트 코퍼스.
        top_k (int): 반환할 최종 결과의 수.

    Returns:
        list: 검색된 문서 객체 리스트.
    """
    vector_results = vectordb.similarity_search(query, k=top_k * 2)  # 벡터 검색 결과는 넉넉하게 가져옴
    keyword_hits = [context for context in keyword_corpus if any(w in context for w in query.split() if len(w) > 1)]

    seen = set()
    merged = []
    # 중복을 제거하면서 벡터 검색 결과와 키워드 검색 결과를 병합
    for doc in vector_results:
        if doc.page_content not in seen:
            merged.append(doc)
            seen.add(doc.page_content)
    for context in keyword_hits:
        if context not in seen:
            from types import SimpleNamespace  # LangChain 문서 형식과 유사하게 맞추기 위함
            merged.append(SimpleNamespace(page_content=context))
            seen.add(context)
    return merged[:top_k]


def extract_sql_and_guide(llm_answer: str) -> (str, str):
    """
    LLM의 답변에서 SQL 쿼리와 가이드 텍스트를 분리합니다.

    Args:
        llm_answer (str): LLM이 생성한 전체 텍스트.

    Returns:
        tuple: (가이드 텍스트, SQL 쿼리).
    """
    answer = llm_answer.replace("```sql", "").replace("```", "").strip()
    # SELECT로 시작하는 SQL 쿼리 부분을 정규식으로 찾음
    sql_match = re.search(r"(SELECT[\s\S]+?)(?:$|\n\n|\Z)", answer, re.IGNORECASE)
    sql = sql_match.group(1).strip() if sql_match else ""
    guide = answer  # SQL 부분을 포함한 전체 텍스트를 가이드로 사용
    return guide, sql


def run_sql_query(sql: str) -> Dict[str, Any]:
    """
    주어진 SQL 쿼리를 Oracle DB에서 실행하고 결과를 반환합니다.

    Args:
        sql (str): 실행할 SELECT 쿼리.

    Returns:
        dict: 실행 성공 여부, 결과 데이터, 오류 메시지 등을 담은 딕셔너리.
    """
    sql = re.sub(r";\s*$", "", sql.strip())  # 마지막 세미콜론 제거
    print("실행 SQL:", sql)

    # SELECT 쿼리가 아닌 경우 실행 방지
    if not sql.lower().startswith("select"):
        return {"success": False, "error": "SELECT 쿼리만 실행 가능합니다."}

    try:
        dsn = cx_Oracle.makedsn(ORACLE_HOST, ORACLE_PORT, service_name=ORACLE_SERVICE)
        with cx_Oracle.connect(ORACLE_USER, ORACLE_PW, dsn) as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql)
                rows = cursor.fetchall()
                col_names = [i[0] for i in cursor.description] if cursor.description else []
                result = [dict(zip(col_names, row)) for row in rows]
                return {"success": True, "result": result, "columns": col_names}
    except Exception as e:
        return {"success": False, "error": str(e)}


def detect_user_intent(msg: str) -> Dict[str, bool]:
    """
    간단한 키워드 매칭으로 사용자의 감성/의도(인사, 긍정, 부정, 욕설)를 탐지합니다.

    Args:
        msg (str): 사용자 메시지.

    Returns:
        dict: 각 의도별 탐지 여부를 bool 값으로 담은 딕셔너리.
    """
    greetings = ['gd', 'ㅎㅇ', '하이', 'hi', 'hello', '안녕']
    positive = ['고마워', '감사', '땡큐', '최고', 'good', 'nice']
    negative = ['싫어', '짜증', '피곤', '힘들', '별로']
    swear = ['씨발', 'ㅅㅂ', 'ㅂㅅ', '병신', '좆', 'fuck', 'shit']
    msg_lower = msg.lower()
    return {
        'greeting': any(word in msg_lower for word in greetings),
        'positive': any(word in msg_lower for word in positive),
        'negative': any(word in msg_lower for word in negative),
        'swear': any(word in msg_lower for word in swear),
    }


# --- 4. LangGraph 상태, 노드, 엣지 정의 ---

# 4.1. 상태(State) 정의: 그래프의 각 노드를 거치며 데이터가 저장되고 전달되는 객체
class GraphState(TypedDict):
    question: str  # 원본 사용자 질문
    chat_history: List[Any]  # 최근 대화 기록
    intent: str  # 분석된 의도
    context: str  # 검색된 컨텍스트 (스키마, FAQ 등)
    guide: str  # LLM에게 제공할 가이드라인
    sql_query: str  # 생성된 SQL 쿼리
    db_result: Dict[str, Any]  # DB 실행 결과
    final_response: str  # 사용자에게 보여줄 최종 답변


# 4.2. 노드(Node) 함수 정의: 그래프의 각 단계를 수행하는 함수

def analyze_intent_node(state: GraphState) -> dict:
    """[노드 1] 사용자 질문의 의도를 분석합니다."""
    print("--- 노드 실행: 의도 분석 ---")
    question = state["question"]
    intent = infer_intent(question)
    return {"intent": intent}


def retrieve_context_node(state: GraphState) -> dict:
    """[노드 2] 분석된 의도를 바탕으로 관련 스키마, 가이드, FAQ를 검색합니다."""
    print("--- 노드 실행: 컨텍스트 검색 ---")
    question = state["question"]
    intent = state["intent"]
    context, guide = load_schema_and_guide(intent)
    faq_corpus = extract_faq_from_context(context)
    # 벡터DB와 키워드 검색을 결합하여 관련성 높은 문서를 찾음
    docs = hybrid_search(question, vectordb, faq_corpus, top_k=3)
    retrieved_context = "\n\n".join([d.page_content for d in docs])
    # 최종적으로 스키마 정보와 검색된 참고 자료를 합쳐서 컨텍스트를 구성
    final_context = f"{context}\n\n---추가 참고 자료---\n{retrieved_context}"
    return {"context": final_context, "guide": guide}


def generate_sql_node(state: GraphState) -> dict:
    """[노드 3] 검색된 컨텍스트를 바탕으로 LLM을 통해 SQL 쿼리를 생성합니다."""
    print("--- 노드 실행: SQL 생성 ---")
    prompt = PromptTemplate(
        input_variables=["context", "guide", "chat_history", "question"],
        template="""[데이터 Context]\n{context}\n\n[분석/SQL 가이드라인]\n{guide}\n\n[대화내용]\n{chat_history}\n\n[사용자 질문]\n{question}"""
    )
    chain = prompt | llm
    result = chain.invoke({
        "context": state["context"],
        "guide": state["guide"],
        "chat_history": state["chat_history"],
        "question": state["question"]
    })
    guide_text, sql = extract_sql_and_guide(result.content)
    # `final_response`에 SQL이 포함된 가이드 텍스트를 임시 저장
    return {"sql_query": sql, "final_response": guide_text.strip()}


def execute_sql_node(state: GraphState) -> dict:
    """[노드 4] 생성된 SQL 쿼리를 DB에서 실행합니다."""
    print("--- 노드 실행: SQL 실행 ---")
    sql = state["sql_query"]
    db_result = run_sql_query(sql)
    return {"db_result": db_result}


def summarize_result_node(state: GraphState) -> dict:
    """[노드 5] SQL 실행 결과를 바탕으로 사용자 친화적인 요약 보고서를 생성합니다."""
    print("--- 노드 실행: 결과 요약 ---")
    db_result = state["db_result"]
    columns = db_result.get("columns", [])
    report_prompt_template = PromptTemplate(
        input_variables=["columns"],
        template="""아래 표는 사용자의 질의에 대한 결과입니다.
딱 한 줄로 결과의 의미만 설명하세요. SQL, 칼럼설명, 예시 등은 답변에 포함하지 마세요.
예시) 20대 남성 골절 진단 환자 명단입니다.

컬럼: {columns}"""
    )
    chain = report_prompt_template | llm
    summary = chain.invoke({"columns": ", ".join(columns)}).content.strip()
    return {"final_response": f"{summary}\n(자세한 정보와 표는 '결과창'에서 확인하세요.)"}


# 4.3. 조건부 엣지(Conditional Edge) 함수 정의: 특정 조건에 따라 다음에 실행할 노드를 결정

def decide_after_generation(state: GraphState) -> str:
    """[분기 1] SQL 생성 후, 쿼리의 존재 여부에 따라 다음 경로를 결정합니다."""
    print("--- 분기: SQL 생성 후 ---")
    if state["sql_query"]:
        print("결과: SQL 존재 -> 실행 노드로 이동")
        return "execute_sql"
    else:
        # LLM이 SQL을 생성하지 않고 바로 답변한 경우, 해당 답변을 최종 결과로 사용하고 종료
        print("결과: SQL 없음 -> 종료")
        return END


def decide_after_execution(state: GraphState) -> str:
    """[분기 2] SQL 실행 후, 성공/실패/결과없음 상태에 따라 다음 경로를 결정합니다."""
    print("--- 분기: SQL 실행 후 ---")
    db_result = state["db_result"]
    if not db_result["success"]:
        print("결과: SQL 실행 실패 -> 오류 처리 후 종료")
        return "handle_db_error"
    elif not db_result.get("result"):
        print("결과: 데이터 없음 -> 결과 없음 처리 후 종료")
        return "handle_no_data"
    else:
        print("결과: 데이터 있음 -> 요약 노드로 이동")
        return "summarize_result"


# 4.4. 예외 처리 노드 정의

def handle_db_error_node(state: GraphState) -> dict:
    """[예외 처리 노드] DB 실행 오류가 발생했을 때, 오류 메시지를 최종 응답으로 설정합니다."""
    print("--- 노드 실행: DB 오류 처리 ---")
    return {"final_response": state["db_result"].get("error", "알 수 없는 데이터베이스 오류입니다.")}


def handle_no_data_node(state: GraphState) -> dict:
    """[예외 처리 노드] 쿼리 결과가 비어있을 때, 데이터가 없다는 메시지를 최종 응답으로 설정합니다."""
    print("--- 노드 실행: 데이터 없음 처리 ---")
    return {"final_response": "조회된 데이터가 없습니다."}


# --- 5. 그래프 빌드 및 컴파일 ---

# StateGraph 객체를 생성하고 상태 클래스를 정의
workflow = StateGraph(GraphState)

# 위에서 정의한 함수들을 그래프의 노드로 추가
workflow.add_node("analyze_intent", analyze_intent_node)
workflow.add_node("retrieve_context", retrieve_context_node)
workflow.add_node("generate_sql", generate_sql_node)
workflow.add_node("execute_sql", execute_sql_node)
workflow.add_node("summarize_result", summarize_result_node)
workflow.add_node("handle_db_error", handle_db_error_node)
workflow.add_node("handle_no_data", handle_no_data_node)

# 그래프의 시작점(진입점) 설정
workflow.set_entry_point("analyze_intent")

# 노드 간의 연결(엣지) 정의
workflow.add_edge("analyze_intent", "retrieve_context")
workflow.add_edge("retrieve_context", "generate_sql")

# 조건부 엣지 정의: 'generate_sql' 노드 실행 후 'decide_after_generation' 함수 결과에 따라 분기
workflow.add_conditional_edges("generate_sql", decide_after_generation)

# 조건부 엣지 정의: 'execute_sql' 노드 실행 후 'decide_after_execution' 함수 결과에 따라 분기
workflow.add_conditional_edges("execute_sql", decide_after_execution)

# 최종 노드에서 END로 연결하여 그래프 종료
workflow.add_edge("summarize_result", END)
workflow.add_edge("handle_db_error", END)
workflow.add_edge("handle_no_data", END)

# 정의된 워크플로우를 실행 가능한 객체로 컴파일
app_graph = workflow.compile()


# --- 6. Flask 라우트(API 엔드포인트) 정의 ---

@app.route('/')
def index():
    """메인 HTML 페이지를 렌더링합니다."""
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    """
    사용자 메시지를 받아 LangGraph를 실행하고, 최종 결과를 JSON 형식으로 반환하는 메인 챗봇 API.
    """
    data = request.json
    user_msg = data['message']

    # 그래프 실행 전 간단한 의도/욕설 필터링
    intent_check = detect_user_intent(user_msg)
    if intent_check['swear']: return jsonify({"report_text": "부적절한 표현은 자제 부탁드립니다."})
    if intent_check['greeting']: return jsonify({"report_text": "안녕하세요! 무엇을 도와드릴까요?"})
    if intent_check['positive']: return jsonify({"report_text": "감사합니다. 더 궁금하신 게 있으신가요?"})
    if intent_check['negative']: return jsonify({"report_text": "많이 지치셨나 봐요. 궁금한 점이 있다면 도와드릴게요."})

    # 그래프 실행을 위한 초기 상태 설정
    initial_state = {
        "question": user_msg,
        "chat_history": data.get('chat_history', [])[-5:]  # 최근 5개 대화기록만 사용
    }

    # LangGraph 실행
    final_state = app_graph.invoke(initial_state)

    # 그래프 최종 상태에서 결과 추출
    db_result = final_state.get('db_result', {})

    # 프론트엔드로 전달할 최종 결과 JSON 구성
    return jsonify({
        "sql": final_state.get('sql_query', ""),
        "db_result": db_result.get("result", [])[:100],  # 프론트엔드 미리보기용 (최대 100개)
        "all_result": db_result.get("result", []),  # 전체 결과 (CSV 다운로드용)
        "db_error": db_result.get("error", None),
        "report_text": final_state.get('final_response', "오류가 발생했습니다."),
        "columns": db_result.get("columns", [])
    })


# --- 7. 기존 사용자 관리 및 기타 라우트 (변경 없음) ---

def get_db_connection():
    """Oracle 데이터베이스 연결 객체를 반환합니다."""
    try:
        dsn = cx_Oracle.makedsn(ORACLE_HOST, ORACLE_PORT, service_name=ORACLE_SERVICE)
        return cx_Oracle.connect(ORACLE_USER, ORACLE_PW, dsn)
    except Exception as e:
        print(f"DB Connection Error: {e}")
        return None


@app.route('/login', methods=['POST'])
def login():
    """사용자 로그인을 처리합니다."""
    data = request.json
    user_id = data.get('user_id')
    user_pw = data.get('user_pw')
    if not user_id or not user_pw:
        return jsonify({"success": False, "message": "ID와 PW를 모두 입력하세요."}), 400
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT USER_SEQ, USER_PW FROM USERS WHERE USER_ID = :1", (user_id,))
                row = cursor.fetchone()
                # 입력된 비밀번호와 DB에 저장된 해시된 비밀번호를 비교
                if row and bcrypt.checkpw(user_pw.encode('utf-8'), row[1].encode('utf-8')):
                    return jsonify({"success": True, "user_seq": row[0]})
                else:
                    return jsonify({"success": False, "message": "로그인 정보가 올바르지 않습니다."})
    except Exception as e:
        return jsonify({"success": False, "message": f"서버 오류: {e}"}), 500


@app.route('/signup', methods=['POST'])
def signup():
    """사용자 회원가입을 처리합니다."""
    data = request.json
    user_id = data.get('user_id')
    user_pw = data.get('user_pw')
    if not user_id or not user_pw:
        return jsonify({"success": False, "message": "ID와 PW를 모두 입력하세요."}), 400
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                # 아이디 중복 확인
                cursor.execute("SELECT COUNT(*) FROM USERS WHERE USER_ID = :1", (user_id,))
                if cursor.fetchone()[0] > 0:
                    return jsonify({"success": False, "message": "이미 존재하는 ID입니다."})

                # 비밀번호를 해시하여 저장
                hashed_pw = bcrypt.hashpw(user_pw.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
                cursor.execute("INSERT INTO USERS (USER_ID, USER_PW) VALUES (:1, :2)", (user_id, hashed_pw))
                conn.commit()
                return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "message": f"서버 오류: {e}"}), 500


@app.route('/change_pw', methods=['POST'])
def change_pw():
    """사용자 비밀번호 변경을 처리합니다."""
    data = request.json
    user_id, old_pw, new_pw = data.get('user_id'), data.get('old_pw'), data.get('new_pw')
    if not all([user_id, old_pw, new_pw]):
        return jsonify({'success': False, 'message': '입력값이 부족합니다.'}), 400
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT USER_PW FROM USERS WHERE USER_ID = :1", (user_id,))
                row = cursor.fetchone()
                # 기존 비밀번호 일치 여부 확인
                if not row or not bcrypt.checkpw(old_pw.encode('utf-8'), row[0].encode('utf-8')):
                    return jsonify({'success': False, 'message': '기존 비밀번호가 일치하지 않습니다.'})

                new_hashed_pw = bcrypt.hashpw(new_pw.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
                cursor.execute("UPDATE USERS SET USER_PW = :1 WHERE USER_ID = :2", (new_hashed_pw, user_id))
                conn.commit()
                return jsonify({'success': True, 'message': '비밀번호가 성공적으로 변경되었습니다.'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'서버 오류: {e}'}), 500


@app.route('/download_csv', methods=['POST'])
def download_csv():
    """JSON 데이터를 받아 CSV 파일로 변환하여 다운로드를 제공합니다."""
    import csv, io
    data = request.json
    result = data.get('data', [])
    columns = data.get('columns', [])

    output = io.StringIO()  # 메모리 상에서 파일처럼 다루기 위함
    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader()
    writer.writerows(result)
    output.seek(0)  # 파일의 처음으로 커서를 이동

    return output.getvalue(), 200, {
        'Content-Type': 'text/csv; charset=utf-8',
        'Content-Disposition': 'attachment; filename="result.csv"'  # 다운로드될 파일 이름 설정
    }


# --- 8. 데이터 분석 탭 관련 스텁(Stub) 라우트들 ---
# 아직 완전히 구현되지 않은 기능들을 위한 임시 엔드포인트입니다.

@app.route('/upload', methods=['POST'])
def upload():
    """파일 업로드 기능을 위한 스텁 라우트."""
    return jsonify({'success': True, 'message': '파일 업로드 성공!(구현 필요)'})


@app.route('/preprocess_data', methods=['POST'])
def preprocess_data():
    """데이터 전처리 기능을 위한 스텁 라우트."""
    return jsonify({'success': True, 'message': '전처리 완료(구현 필요)'})


@app.route('/analyze', methods=['POST'])
def analyze():
    """데이터 분석 실행을 위한 스텁 라우트."""
    return jsonify({'success': True, 'results': []})


# --- 애플리케이션 실행 ---
if __name__ == '__main__':
    # 파일 업로드를 위한 'uploads' 폴더가 없으면 생성
    if not os.path.exists('./uploads'):
        os.makedirs('./uploads')

    # Flask 개발 서버 실행
    app.run(host='0.0.0.0', port=5001, debug=True)