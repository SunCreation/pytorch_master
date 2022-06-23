
import os 
import argparse

# parser 정의
parser = argparse.ArgumentParser(description='Argparse Tutorial')
# add_argument()를 통해 argument의 이름, 타입, 기본 값, 도움말을 정의할수 있다.
parser.add_argument('-k','--korean', type=int, default=0, help="Score of korean")
parser.add_argument('-m','--mathematics', type=int, default=0,help="Score of mathematcis")
parser.add_argument('-e', '--english', type=int, default=0, help = "Score of english")

# add_argment()함수를 호출하면 parser인스턴스 내부에 해당 ㅣㅇ름을 가지는 멤버 변수를 생성
# parse_arg()를 통해 프로그램 실행시 parser가 실행되도록 합니다.
args = parser.parse_args()

# subject_info = {'korean': args.n}
def average(args):
    total_score = 0
    total_score += args.korean
    total_score += args.mathematics
    total_score += args.english
    print(total_score/3)

average(args)