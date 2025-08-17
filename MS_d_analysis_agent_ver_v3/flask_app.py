from flask import Flask, request, jsonify, render_template, session
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import io
import os
import operator
from typing import Dict, List, TypedDict, Annotated
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg
import uuid
import threading 

# LangChain 관련 라이브러리
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# 데이터 처리 및 통계 계산을 위한 라이브러리
from scipy.stats import pearsonr, ttest_ind, f_oneway, chi2_contingency
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # 실제 운영에서는 안전한 키로 변경하세요

# 백그라운드 작업의 상태와 결과를 저장할 인메모리 딕셔너리
tasks = {}


# ==============================================================================
# 도구(Tools) 정의
# ==============================================================================
@tool
def descriptive_analysis(data_json: str, var: str) -> Dict:
    """
    단일 변수에 대한 기술 통계(빈도, 분포)를 분석합니다. 본격적인 분석 전 데이터의 특징을 파악하기 위해 사용합니다.
    :param data_json: 분석할 데이터프레임의 JSON 문자열.
    :param var: 분석할 변수 이름.
    """
    try:
        data = pd.read_json(io.StringIO(data_json))
        clean_data = data[[var]].dropna()

        if pd.api.types.is_numeric_dtype(clean_data[var]):
            stats_df = clean_data[var].describe().round(2).to_dict()
            stats_df = stats_df.replace({np.nan: None})
            stats = stats_df.to_dict()
            return {
                'tool': 'descriptive_analysis',
                'variable': var,
                'variable_type': 'numeric',
                'results': {'statistics': stats},
                'plot_data': {
                    'type': 'histogram',
                    'data': clean_data[var].tolist(),
                    'title': f'Distribution of {var}',
                    'xlabel': var
                }
            }
        else:
            counts = clean_data[var].value_counts().to_dict()
            percentages = (clean_data[var].value_counts(normalize=True) * 100).round(2).to_dict()
            return {
                'tool': 'descriptive_analysis',
                'variable': var,
                'variable_type': 'categorical',
                'results': {'counts': counts, 'percentages': percentages},
                'plot_data': {
                    'type': 'bar',
                    'data': counts,
                    'title': f'Frequency of {var}',
                    'xlabel': var
                }
            }
    except Exception as e:
        return {'error': f'기술 통계 분석 중 오류 발생: {str(e)}'}


@tool
def correlation_analysis(data_json: str, var1: str, var2: str) -> Dict:
    """
    두 수치형 변수 간의 피어슨 상관관계를 분석합니다.
    :param data_json: 분석할 데이터프레임의 JSON 문자열.
    :param var1: 첫 번째 변수 이름.
    :param var2: 두 번째 변수 이름.
    """
    try:
        data = pd.read_json(io.StringIO(data_json))
        df_copy = data[[var1, var2]].copy()
        df_copy[var1] = pd.to_numeric(df_copy[var1], errors='coerce')
        df_copy[var2] = pd.to_numeric(df_copy[var2], errors='coerce')
        clean_data = df_copy.dropna()
        if len(clean_data) < 3:
            return {'error': '상관분석을 위한 유효 데이터가 부족합니다.'}
        corr_coef, p_value = pearsonr(clean_data[var1], clean_data[var2])
        return {
            'tool': 'correlation_analysis',
            'variables': [var1, var2],
            'results': {'correlation_coefficient': corr_coef, 'p_value': p_value},
            'plot_data': {
                'type': 'scatter',
                'x': clean_data[var1].tolist(),
                'y': clean_data[var2].tolist(),
                'title': f'Correlation: {var1} vs {var2}',
                'xlabel': var1,
                'ylabel': var2
            }
        }
    except Exception as e:
        return {'error': f'상관분석 중 오류 발생: {str(e)}'}


@tool
def group_comparison_analysis(data_json: str, cat_var: str, num_var: str) -> Dict:
    """
    범주형 변수에 따른 수치형 변수의 평균을 비교합니다 (t-test 또는 ANOVA).
    :param data_json: 분석할 데이터프레임의 JSON 문자열.
    :param cat_var: 그룹을 나누는 범주형 변수.
    :param num_var: 평균을 비교할 수치형 변수.
    """
    try:
        data = pd.read_json(io.StringIO(data_json))
        df_copy = data[[cat_var, num_var]].copy()
        df_copy[num_var] = pd.to_numeric(df_copy[num_var], errors='coerce')
        clean_data = df_copy.dropna()
        groups = [group[num_var].values for name, group in clean_data.groupby(cat_var)]
        if len(groups) < 2:
            return {'error': '그룹 비교를 위한 유효 데이터가 부족합니다 (최소 2개 그룹 필요).'}
        test_name, stat, pval = ("t-test", *ttest_ind(groups[0], groups[1])) if len(groups) == 2 else ("ANOVA",
                                                                                                       *f_oneway(
                                                                                                           *groups))
        return {
            'tool': 'group_comparison_analysis',
            'variables': [cat_var, num_var],
            'results': {'test_statistic': stat, 'p_value': pval, 'test_type': test_name},
            'plot_data': {
                'type': 'boxplot',
                'data': clean_data.to_dict('list'),
                'x': cat_var,
                'y': num_var,
                'title': f'{test_name}: {num_var} by {cat_var}'
            }
        }
    except Exception as e:
        return {'error': f'그룹 비교 분석 중 오류 발생: {str(e)}'}


@tool
def simple_linear_regression_analysis(data_json: str, independent_var: str, dependent_var: str) -> Dict:
    """
    단순 선형 회귀분석을 수행하여 하나의 변수로 다른 변수를 예측하는 모델을 만듭니다.
    :param data_json: 분석할 데이터프레임의 JSON 문자열.
    :param independent_var: 예측에 사용할 독립 변수 (x축, 수치형).
    :param dependent_var: 예측하려는 종속 변수 (y축, 수치형).
    """
    try:
        data = pd.read_json(io.StringIO(data_json))
        df_copy = data[[independent_var, dependent_var]].copy()
        df_copy[independent_var] = pd.to_numeric(df_copy[independent_var], errors='coerce')
        df_copy[dependent_var] = pd.to_numeric(df_copy[dependent_var], errors='coerce')
        clean_data = df_copy.dropna()
        if len(clean_data) < 3:
            return {'error': '회귀분석을 위한 유효 데이터가 부족합니다.'}
        X, y = clean_data[[independent_var]], clean_data[dependent_var]
        model = LinearRegression().fit(X, y)
        y_pred = model.predict(X)
        r_squared, coefficient, intercept = r2_score(y, y_pred), model.coef_[0], model.intercept_
        return {
            'tool': 'simple_linear_regression_analysis',
            'variables': {'independent': independent_var, 'dependent': dependent_var},
            'results': {
                'r_squared': r_squared,
                'coefficient': coefficient,
                'intercept': intercept,
                'equation': f'{dependent_var} = {coefficient:.2f} * {independent_var} + {intercept:.2f}'
            },
            'plot_data': {
                'type': 'regression',
                'x': X.values.flatten().tolist(),
                'y': y.values.flatten().tolist(),
                'y_pred': y_pred.tolist(),
                'title': f'Regression: {independent_var} vs {dependent_var}',
                'xlabel': independent_var,
                'ylabel': dependent_var
            }
        }
    except Exception as e:
        return {'error': f'회귀분석 중 오류 발생: {str(e)}'}


@tool
def chi_squared_test(data_json: str, var1: str, var2: str) -> Dict:
    """
    두 범주형 변수 간의 연관성이 있는지 카이제곱 독립성 검정을 수행합니다.
    :param data_json: 분석할 데이터프레임의 JSON 문자열.
    :param var1: 첫 번째 범주형 변수.
    :param var2: 두 번째 범주형 변수.
    """
    try:
        data = pd.read_json(io.StringIO(data_json))
        df_copy = data[[var1, var2]].dropna()
        if len(df_copy) < 5:
            return {'error': '카이제곱 검정을 위한 데이터가 부족합니다.'}
        contingency_table = pd.crosstab(df_copy[var1], df_copy[var2])
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        return {
            'tool': 'chi_squared_test',
            'variables': [var1, var2],
            'results': {'chi2_statistic': chi2, 'p_value': p, 'degrees_of_freedom': dof},
            'contingency_table': contingency_table.to_dict()
        }
    except Exception as e:
        return {'error': f'카이제곱 검정 중 오류 발생: {str(e)}'}


# ==============================================================================
# LangGraph 상태(State) 정의
# ==============================================================================
class AgentState(TypedDict):
    data_json: str
    data_profile: str
    plan: List[Dict]
    report: str
    messages: Annotated[list, operator.add]


# ==============================================================================
# 에이전트 및 노드(Nodes) 정의
# ==============================================================================
class ToolCall(BaseModel):
    tool_name: str = Field(
        description="사용할 도구의 이름. 반드시 ['descriptive_analysis', 'correlation_analysis', 'group_comparison_analysis', 'simple_linear_regression_analysis', 'chi_squared_test'] 중 하나여야 합니다.")
    tool_args: Dict = Field(description="도구에 전달할 인자 딕셔너리.")


class AnalysisPlan(BaseModel):
    tool_calls: List[ToolCall] = Field(description="실행할 통계 분석 도구 호출 목록")


def create_planner_agent(llm):
    parser = PydanticOutputParser(pydantic_object=AnalysisPlan)
    prompt_template = """당신은 데이터 분석 계획을 수립하는 전문 데이터 과학자입니다.
주어진 데이터 프로필을 기반으로, 데이터의 핵심적인 특징을 파악하기 위해 수행해야 할 통계 분석 도구들의 목록을 제안해주세요.
먼저 주요 변수들에 대해 'descriptive_analysis'를 실행하여 데이터의 기본적인 분포와 특징을 파악하는 것을 우선으로 고려하세요.
그 후, 변수들의 관계를 더 깊이 파악할 수 있는 다른 분석들을 계획하세요. 각 분석은 반드시 사용 가능한 도구 중 하나여야 합니다.

사용 가능한 도구:
- descriptive_analysis(var: str): 단일 변수의 분포(수치형)나 빈도(범주형)를 분석합니다. 가장 먼저 실행되어야 합니다.
- correlation_analysis(var1: str, var2: str): 두 수치형 변수 간의 상관관계 분석.
- group_comparison_analysis(cat_var: str, num_var: str): 범주형 변수에 따른 수치형 변수 평균 비교.
- simple_linear_regression_analysis(independent_var: str, dependent_var: str): 하나의 수치형 변수로 다른 수치형 변수를 예측하는 모델 생성.
- chi_squared_test(var1: str, var2: str): 두 범주형 변수 간의 연관성 검정.

{format_instructions}

데이터 프로필:
{data_profile}
"""

    def planner_node(state: AgentState):
        profile_str = json.dumps(state['data_profile'], indent=2, ensure_ascii=False)
        prompt = prompt_template.format(data_profile=profile_str, format_instructions=parser.get_format_instructions())
        response = llm.invoke(prompt)
        try:
            parsed_plan = parser.parse(response.content)
            plan = [call.dict() for call in parsed_plan.tool_calls]
            return {"plan": plan}
        except Exception as e:
            return {"plan": []}

    return planner_node


tools = [descriptive_analysis, correlation_analysis, group_comparison_analysis, simple_linear_regression_analysis,
         chi_squared_test]
statistics_expert_node = ToolNode(tools)


def create_reporter_agent(llm):
    prompt = """당신은 데이터 분석 결과를 종합하여 최종 보고서를 작성하는 전문 애널리스트입니다.
아래는 여러 통계 도구를 실행하여 얻은 결과들의 목록입니다. 각 결과를 해석하고, 통계적 유의성을 설명하며,
데이터로부터 얻을 수 있는 전반적인 인사이트를 종합하여 하나의 완성된 보고서를 한국어 마크다운 형식으로 작성해주세요.

분석 결과 목록:
{analysis_results}
"""

    def reporter_node(state: AgentState):
        tool_outputs = [json.loads(msg.content) for msg in state['messages'] if isinstance(msg, ToolMessage)]
        results_str = json.dumps(tool_outputs, indent=2, ensure_ascii=False)
        response = llm.invoke(prompt.format(analysis_results=results_str))
        return {"report": response.content}

    return reporter_node


def prepare_next_step_node(state: AgentState):
    if len(state.get('plan', [])) > sum(isinstance(msg, ToolMessage) for msg in state['messages']):
        next_tool_index = sum(isinstance(msg, ToolMessage) for msg in state['messages'])
        next_tool_call = state['plan'][next_tool_index]
        return {"messages": [AIMessage(content="", tool_calls=[
            {"id": f"tool_{next_tool_call['tool_name']}_{next_tool_index}", "name": next_tool_call['tool_name'],
             "args": {"data_json": state['data_json'], **next_tool_call['tool_args']}}])]}
    else:
        return {}


def route_after_prepare(state: AgentState):
    if len(state.get('plan', [])) > sum(isinstance(msg, ToolMessage) for msg in state['messages']):
        return "statistics_expert"
    else:
        return "reporter"


# ==============================================================================
# 그래프(Graph) 생성 및 연결
# ==============================================================================

def create_graph(llm, for_planning_only=False):
    graph_builder = StateGraph(AgentState)

    if for_planning_only:
        planner_node = create_planner_agent(llm)
        graph_builder.add_node("planner", planner_node)
        graph_builder.set_entry_point("planner")
        graph_builder.add_edge("planner", END)
    else:
        reporter_node = create_reporter_agent(llm)
        graph_builder.add_node("prepare_next_step", prepare_next_step_node)
        graph_builder.add_node("statistics_expert", statistics_expert_node)
        graph_builder.add_node("reporter", reporter_node)
        graph_builder.set_entry_point("prepare_next_step")
        graph_builder.add_edge("statistics_expert", "prepare_next_step")
        graph_builder.add_conditional_edges("prepare_next_step", route_after_prepare,
                                            {"statistics_expert": "statistics_expert", "reporter": "reporter"})
        graph_builder.add_edge("reporter", END)

    return graph_builder.compile()


# ==============================================================================
# 차트 생성 유틸리티 함수
# ==============================================================================

def create_plot_base64(plot_data: Dict) -> str:
    """차트를 생성하고 base64 인코딩된 이미지 문자열을 반환"""
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_type = plot_data.get('type')

    try:
        if plot_type == 'scatter':
            ax.scatter(plot_data['x'], plot_data['y'], alpha=0.6)
        elif plot_type == 'boxplot':
            data_df = pd.DataFrame(plot_data['data'])
            data_df.boxplot(column=plot_data['y'], by=plot_data['x'], ax=ax)
            plt.suptitle('')
        elif plot_type == 'histogram':
            ax.hist(plot_data['data'], bins=30, alpha=0.7)
        elif plot_type == 'bar':
            bar_data = plot_data['data']
            if len(bar_data) > 20:
                bar_data = dict(list(bar_data.items())[:20])
                ax.set_title(f"Top 20 {plot_data.get('title', '')}")
            else:
                ax.set_title(plot_data.get('title', ''))
            ax.bar(bar_data.keys(), bar_data.values())
            plt.xticks(rotation=45, ha='right')
        elif plot_type == 'regression':
            ax.scatter(plot_data['x'], plot_data['y'], alpha=0.6, label='Actual Data')
            ax.plot(plot_data['x'], plot_data['y_pred'], color='red', linewidth=2, label='Regression Line')
            ax.legend()

        ax.set_title(plot_data.get('title', ''))
        ax.set_xlabel(plot_data.get('xlabel', ''))
        ax.set_ylabel(plot_data.get('ylabel', ''))
        plt.tight_layout()

        buffer = io.BytesIO()
        canvas = FigureCanvasAgg(fig)
        canvas.print_png(buffer)
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close(fig)

        graphic = base64.b64encode(image_png)
        graphic = graphic.decode('utf-8')
        return graphic

    except Exception as e:
        plt.close(fig)
        return None


# ==============================================================================
# 백그라운드 분석 실행 함수
# ==============================================================================
def run_analysis_in_background(task_id, api_key, selected_plan, df_json):
    """실제 분석을 수행하고 결과를 tasks 딕셔너리에 저장하는 함수"""
    try:
        global tasks
        tasks[task_id]['status'] = 'running'

        os.environ["OPENAI_API_KEY"] = api_key
        llm = ChatOpenAI(model="gpt-4o")

        execution_graph = create_graph(llm, for_planning_only=False)
        initial_state = {
            "data_json": df_json,
            "plan": selected_plan,
            "messages": []
        }

        execution_results = []
        final_report = ""

        events = execution_graph.stream(initial_state, config={"recursion_limit": 100})
        for event in events:
            for node_name, state_update in event.items():
                if state_update is None:
                    continue
                if node_name == "statistics_expert" and state_update.get('messages'):
                    last_message = state_update['messages'][-1]
                    if isinstance(last_message, ToolMessage):
                        result = json.loads(last_message.content)
                        if 'plot_data' in result and result['plot_data']:
                            chart_base64 = create_plot_base64(result['plot_data'])
                            if chart_base64:
                                result['chart'] = chart_base64
                        execution_results.append(result)
                elif node_name == "reporter" and 'report' in state_update:
                    final_report = state_update['report']

        # 작업 완료 후 결과 저장
        tasks[task_id].update({
            'status': 'completed',
            'results': execution_results,
            'report': final_report,
            'message': f'{len(execution_results)}개의 분석이 완료되었습니다.'
        })

    except Exception as e:
        # 오류 발생 시 상태 업데이트
        tasks[task_id].update({
            'status': 'failed',
            'message': f'분석 실행 중 오류가 발생했습니다: {str(e)}'
        })


# ==============================================================================
# Flask 라우트 정의
# ==============================================================================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    try:
        file = request.files['file']
        if file and file.filename.endswith('.csv'):
            df = pd.read_csv(file)
            df = df.replace({np.nan: None})
            session['df_json'] = df.to_json(orient='records')
            session['df_info'] = {
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': {k: str(v) for k, v in df.dtypes.to_dict().items()}
            }
            preview_data = df.head().to_dict('records')
            return jsonify({
                'success': True,
                'shape': df.shape,
                'columns': list(df.columns),
                'preview': preview_data,
                'message': f'파일이 성공적으로 업로드되었습니다. ({df.shape[0]}행, {df.shape[1]}열)'
            })
        else:
            return jsonify({'success': False, 'message': 'CSV 파일만 업로드 가능합니다.'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'파일 업로드 중 오류가 발생했습니다: {str(e)}'})


@app.route('/generate_plan', methods=['POST'])
def generate_plan():
    try:
        data = request.json
        api_key = data.get('api_key')
        if not api_key:
            return jsonify({'success': False, 'message': 'OpenAI API 키가 필요합니다.'})
        if 'df_json' not in session:
            return jsonify({'success': False, 'message': '먼저 CSV 파일을 업로드해주세요.'})

        os.environ["OPENAI_API_KEY"] = api_key
        llm = ChatOpenAI(model="gpt-4o")

        df = pd.read_json(io.StringIO(session['df_json']))
        profile = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': {k: str(v) for k, v in df.dtypes.to_dict().items()},
            'numeric_columns': df.select_dtypes(include=np.number).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist()
        }
        planning_graph = create_graph(llm, for_planning_only=True)
        result = planning_graph.invoke({"data_profile": profile})
        plan = result.get('plan', [])
        session['analysis_plan'] = plan

        return jsonify({
            'success': True,
            'plan': plan,
            'message': f'{len(plan)}개의 분석이 제안되었습니다.'
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'분석 계획 생성 중 오류가 발생했습니다: {str(e)}'})


# 👈 [수정] /execute_analysis 라우트 수정
@app.route('/execute_analysis', methods=['POST'])
def execute_analysis():
    data = request.json
    api_key = data.get('api_key')
    selected_indices = data.get('selected_analyses', [])

    if not api_key:
        return jsonify({'success': False, 'message': 'OpenAI API 키가 필요합니다.'})
    if 'df_json' not in session or 'analysis_plan' not in session:
        return jsonify({'success': False, 'message': '먼저 CSV 파일을 업로드하고 분석 계획을 생성해주세요.'})

    full_plan = session['analysis_plan']
    selected_plan = [full_plan[i] for i in selected_indices if i < len(full_plan)]

    if not selected_plan:
        return jsonify({'success': False, 'message': '실행할 분석을 선택해주세요.'})

    # 작업 ID 생성 및 초기 상태 설정
    task_id = str(uuid.uuid4())
    tasks[task_id] = {'status': 'pending'}

    # 백그라운드 스레드에서 분석 실행
    thread = threading.Thread(target=run_analysis_in_background, args=(
        task_id,
        api_key,
        selected_plan,
        session['df_json']
    ))
    thread.start()

    # 작업 ID를 클라이언트에 즉시 반환
    return jsonify({'success': True, 'task_id': task_id})


#작업 상태를 반환하는 새로운 라우트 추가
@app.route('/get_task_status/<task_id>')
def get_task_status(task_id):
    """작업 ID를 기반으로 현재 작업 상태와 결과를 반환"""
    task = tasks.get(task_id, {'status': 'not_found', 'message': '작업을 찾을 수 없습니다.'})
    return jsonify(task)


if __name__ == '__main__':
    app.run(debug=True)
