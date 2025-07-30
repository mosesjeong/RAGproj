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

# LangGraph 관련 라이브러리 임포트
from langgraph.graph import StateGraph, END

# --- 1. 설정: 환경 변수 및 기본 설정 ---
# Langsmith 및 OpenAI API 키 설정
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"] = "본인 api key"
os.environ["LANGSMITH_PROJECT"] = "YB_TEST_CODE_V6"
os.environ["OPENAI_API_KEY"] = "본인 api key"

app = Flask(__name__)
CORS(app)

# Oracle DB 정보
ORACLE_USER = os.getenv("ORACLE_USER", "본인DB정보")
ORACLE_PW = os.getenv("ORACLE_PW", "본인DB정보")
ORACLE_HOST = os.getenv("ORACLE_HOST", "본인DB정보")
ORACLE_PORT = int(os.getenv("ORACLE_PORT", "본인DB정보"))
ORACLE_SERVICE = os.getenv("ORACLE_SERVICE", "본인DB정보")

# --- 2. LangChain 및 기타 전역 객체 초기화 ---
llm = ChatOpenAI(model="gpt-4o", temperature=0)
embedding = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
vectordb = Chroma(persist_directory="./chroma_db", embedding_function=embedding)

with open('guide_map.json', encoding='utf-8') as f:
    GUIDE_MAP = json.load(f)


# --- 3. 기존 헬퍼 함수들 (변경 없음) ---
# 이 함수들은 LangGraph의 노드 안에서 재사용됩니다.
def infer_intent(question):
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


def load_schema_and_guide(intent):
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


def extract_faq_from_context(context):
    faq_list = []
    faq_pairs = re.findall(r"Q[:：](.*?)\nA[:：](.*?)(?=\nQ[:：]|\Z)", context, re.DOTALL)
    for q, a in faq_pairs:
        faq_list.append("Q:" + q.strip() + "\nA:" + a.strip())
    return faq_list


def hybrid_search(query, vectordb, keyword_corpus, top_k=3):
    vector_results = vectordb.similarity_search(query, k=top_k * 2)
    keyword_hits = [context for context in keyword_corpus if any(w in context for w in query.split() if len(w) > 1)]
    seen = set()
    merged = []
    for doc in vector_results:
        if doc.page_content not in seen:
            merged.append(doc)
            seen.add(doc.page_content)
    for context in keyword_hits:
        if context not in seen:
            from types import SimpleNamespace
            merged.append(SimpleNamespace(page_content=context))
            seen.add(context)
    return merged[:top_k]


def extract_sql_and_guide(llm_answer):
    answer = llm_answer.replace("```sql", "").replace("```", "").strip()
    sql_match = re.search(r"(SELECT[\s\S]+?)(?:$|\n\n|\Z)", answer, re.IGNORECASE)
    sql = sql_match.group(1).strip() if sql_match else ""
    guide = answer
    return guide, sql


def run_sql_query(sql):
    sql = re.sub(r";\s*$", "", sql.strip())
    print("실행 SQL:", sql)
    if not sql.lower().startswith("select"): return {"success": False, "error": "SELECT 쿼리만 실행 가능합니다."}
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


def detect_user_intent(msg):
    # (기존 함수 내용과 동일)
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

# 4.1. 상태(State) 정의
class GraphState(TypedDict):
    question: str
    chat_history: List[Any]
    intent: str
    context: str
    guide: str
    sql_query: str
    db_result: Dict[str, Any]
    final_response: str


# 4.2. 노드(Node) 함수 정의
def analyze_intent_node(state: GraphState) -> dict:
    print("--- 노드 실행: 의도 분석 ---")
    question = state["question"]
    intent = infer_intent(question)
    return {"intent": intent}


def retrieve_context_node(state: GraphState) -> dict:
    print("--- 노드 실행: 컨텍스트 검색 ---")
    question = state["question"]
    intent = state["intent"]
    context, guide = load_schema_and_guide(intent)
    faq_corpus = extract_faq_from_context(context)
    docs = hybrid_search(question, vectordb, faq_corpus, top_k=3)
    retrieved_context = "\n\n".join([d.page_content for d in docs])
    final_context = f"{context}\n\n---추가 참고 자료---\n{retrieved_context}"
    return {"context": final_context, "guide": guide}


def generate_sql_node(state: GraphState) -> dict:
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
    return {"sql_query": sql, "final_response": guide_text.strip()}


def execute_sql_node(state: GraphState) -> dict:
    print("--- 노드 실행: SQL 실행 ---")
    sql = state["sql_query"]
    db_result = run_sql_query(sql)
    return {"db_result": db_result}


def summarize_result_node(state: GraphState) -> dict:
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


# 4.3. 조건부 엣지(Conditional Edge) 함수 정의
def decide_after_generation(state: GraphState) -> str:
    print("--- 분기: SQL 생성 후 ---")
    if state["sql_query"]:
        print("결과: SQL 존재 -> 실행 노드로 이동")
        return "execute_sql"
    else:
        print("결과: SQL 없음 -> 종료")
        return END


def decide_after_execution(state: GraphState) -> str:
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


# 4.4. 에러 및 최종 처리 노드 추가
def handle_db_error_node(state: GraphState) -> dict:
    print("--- 노드 실행: DB 오류 처리 ---")
    return {"final_response": state["db_result"].get("error", "알 수 없는 데이터베이스 오류입니다.")}


def handle_no_data_node(state: GraphState) -> dict:
    print("--- 노드 실행: 데이터 없음 처리 ---")
    return {"final_response": "조회된 데이터가 없습니다."}


# --- 5. 그래프 빌드 및 컴파일 ---
workflow = StateGraph(GraphState)

workflow.add_node("analyze_intent", analyze_intent_node)
workflow.add_node("retrieve_context", retrieve_context_node)
workflow.add_node("generate_sql", generate_sql_node)
workflow.add_node("execute_sql", execute_sql_node)
workflow.add_node("summarize_result", summarize_result_node)
workflow.add_node("handle_db_error", handle_db_error_node)
workflow.add_node("handle_no_data", handle_no_data_node)

workflow.set_entry_point("analyze_intent")
workflow.add_edge("analyze_intent", "retrieve_context")
workflow.add_edge("retrieve_context", "generate_sql")
workflow.add_conditional_edges("generate_sql", decide_after_generation)
workflow.add_conditional_edges("execute_sql", decide_after_execution)

workflow.add_edge("summarize_result", END)
workflow.add_edge("handle_db_error", END)
workflow.add_edge("handle_no_data", END)

app_graph = workflow.compile()


# --- 6. Flask 라우트 정의 ---

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_msg = data['message']

    # 그래프 실행 전 간단한 의도/욕설 필터링 (기존 로직 유지)
    intent_check = detect_user_intent(user_msg)
    if intent_check['swear']: return jsonify({"report_text": "부적절한 표현은 자제 부탁드립니다."})
    if intent_check['greeting']: return jsonify({"report_text": "안녕하세요! 무엇을 도와드릴까요?"})
    if intent_check['positive']: return jsonify({"report_text": "감사합니다. 더 궁금하신 게 있으신가요?"})
    if intent_check['negative']: return jsonify({"report_text": "많이 지치셨나 봐요. 궁금한 점이 있다면 도와드릴게요."})

    # 그래프 실행
    initial_state = {
        "question": user_msg,
        "chat_history": data.get('chat_history', [])[-5:]
    }

    # LangGraph 실행!
    final_state = app_graph.invoke(initial_state)

    # 그래프 최종 상태에서 결과 추출하여 반환
    db_result = final_state.get('db_result', {})

    return jsonify({
        "sql": final_state.get('sql_query', ""),
        "db_result": db_result.get("result", [])[:100],  # 프론트엔드 미리보기용
        "all_result": db_result.get("result", []),  # 전체 결과
        "db_error": db_result.get("error", None),
        "report_text": final_state.get('final_response', "오류가 발생했습니다."),
        "columns": db_result.get("columns", [])
    })


# --- 기존 사용자 관리 및 기타 라우트 (변경 없음) ---
# (이하 /login, /signup, /change_pw, /download_csv 등 기존 함수들)
def get_db_connection():
    try:
        dsn = cx_Oracle.makedsn(ORACLE_HOST, ORACLE_PORT, service_name=ORACLE_SERVICE)
        return cx_Oracle.connect(ORACLE_USER, ORACLE_PW, dsn)
    except Exception as e:
        print(f"DB Connection Error: {e}")
        return None


@app.route('/login', methods=['POST'])
def login():
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
                if row and bcrypt.checkpw(user_pw.encode('utf-8'), row[1].encode('utf-8')):
                    return jsonify({"success": True, "user_seq": row[0]})
                else:
                    return jsonify({"success": False, "message": "로그인 정보가 올바르지 않습니다."})
    except Exception as e:
        return jsonify({"success": False, "message": f"서버 오류: {e}"}), 500


@app.route('/signup', methods=['POST'])
def signup():
    data = request.json
    user_id = data.get('user_id')
    user_pw = data.get('user_pw')
    if not user_id or not user_pw:
        return jsonify({"success": False, "message": "ID와 PW를 모두 입력하세요."}), 400
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM USERS WHERE USER_ID = :1", (user_id,))
                if cursor.fetchone()[0] > 0:
                    return jsonify({"success": False, "message": "이미 존재하는 ID입니다."})
                hashed_pw = bcrypt.hashpw(user_pw.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
                cursor.execute("INSERT INTO USERS (USER_ID, USER_PW) VALUES (:1, :2)", (user_id, hashed_pw))
                conn.commit()
                return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "message": f"서버 오류: {e}"}), 500


@app.route('/change_pw', methods=['POST'])
def change_pw():
    data = request.json
    user_id, old_pw, new_pw = data.get('user_id'), data.get('old_pw'), data.get('new_pw')
    if not all([user_id, old_pw, new_pw]):
        return jsonify({'success': False, 'message': '입력값이 부족합니다.'}), 400
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT USER_PW FROM USERS WHERE USER_ID = :1", (user_id,))
                row = cursor.fetchone()
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
    import csv, io
    data = request.json
    result = data.get('data', [])
    columns = data.get('columns', [])
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader()
    writer.writerows(result)
    output.seek(0)
    return output.getvalue(), 200, {
        'Content-Type': 'text/csv; charset=utf-8',
        'Content-Disposition': 'attachment; filename="result.csv"'
    }


# --- 데이터 분석 탭 관련 스텁(Stub) 라우트들 ---
@app.route('/upload', methods=['POST'])
def upload():
    return jsonify({'success': True, 'message': '파일 업로드 성공!(구현 필요)'})


@app.route('/preprocess_data', methods=['POST'])
def preprocess_data():
    return jsonify({'success': True, 'message': '전처리 완료(구현 필요)'})


@app.route('/analyze', methods=['POST'])
def analyze():
    return jsonify({'success': True, 'results': []})


if __name__ == '__main__':
    if not os.path.exists('./uploads'):
        os.makedirs('./uploads')
    app.run(host='0.0.0.0', port=5001, debug=True)
