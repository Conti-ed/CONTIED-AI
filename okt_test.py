import jpype
from konlpy.tag import Okt

# JVM 경로 수동 설정
jvm_path = jpype.getDefaultJVMPath()

# .jar 파일 경로 설정
classpath = (
    r"C:\projects\konlpy\java\open-korean-text-2.1.0.jar;"
    r"C:\projects\konlpy\java\scala-compiler-2.12.12.jar;"
    r"C:\projects\konlpy\java\scala-library-2.12.12.jar;"
    r"C:\projects\konlpy\java\scala-reflect-2.12.12.jar;"
    r"C:\projects\konlpy\java\scala-parser-combinators_2.12-1.1.2.jar;"
    r"C:\projects\konlpy\java\twitter-text-2.0.8.jar"
)

# JVM 시작 시 classpath를 명시적으로 설정
if not jpype.isJVMStarted():
    jpype.startJVM(jvm_path, f"-Djava.class.path={classpath}", "-ea")

# Okt 객체 생성 및 테스트
try:
    okt = Okt()
    print("Okt 객체 생성 성공!")

    # 테스트 문장 형태소 분석
    print("형태소 분석 결과:", okt.morphs("안녕하세요, 형태소 분석기 테스트입니다."))
    print("명사 추출 결과:", okt.nouns("안녕하세요, 형태소 분석기 테스트입니다."))
    print("구 추출 결과:", okt.phrases("안녕하세요, 형태소 분석기 테스트입니다."))
except Exception as e:
    print(f"Okt 객체 생성 중 오류 발생: {e}")
