import os
import glob
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# 1. 문서 로드 및 청킹 설정
def build_database():
    """
    현재 폴더의 모든 .txt 파일을 읽어 청킹하고 Chroma DB를 생성합니다.
    """
    print("데이터베이스 구축을 시작합니다...")

    # 모든 .txt 파일 경로 가져오기
    all_txt_files = glob.glob("*.txt")
    if not all_txt_files:
        print("오류: 현재 폴더에 .txt 파일이 없습니다.")
        return

    print(f"총 {len(all_txt_files)}개의 .txt 파일을 찾았습니다.")

    # 텍스트 스플리터 초기화
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len
    )

    # 모든 파일의 내용을 담을 리스트
    all_docs = []

    # 각 파일을 읽고 청크 생성
    for file_path in all_txt_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                chunks = text_splitter.split_text(text)
                # 각 청크에 출처(source) 메타데이터 추가
                for chunk in chunks:
                    all_docs.append((chunk, {"source": os.path.basename(file_path)}))
            print(f" - '{file_path}' 처리 완료.")
        except Exception as e:
            print(f" - '{file_path}' 처리 중 오류 발생: {e}")

    if not all_docs:
        print("오류: 유효한 텍스트 청크를 생성하지 못했습니다.")
        return

    print(f"\n총 {len(all_docs)}개의 청크를 생성했습니다. 벡터 임베딩 및 DB 저장을 시작합니다...")

    # 임베딩 모델 로드
    embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

    # Chroma DB 생성 및 저장
    # from_texts를 사용하여 튜플 리스트를 직접 처리
    texts, metadatas = zip(*all_docs)
    vectordb = Chroma.from_texts(
        texts=list(texts),
        embedding=embedding_model,
        metadatas=list(metadatas),
        persist_directory="./chroma_db"
    )

    print("\n✅ 데이터베이스 구축 및 저장이 완료되었습니다. ('./chroma_db' 폴더 확인)")


if __name__ == "__main__":
    build_database()