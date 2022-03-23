
'''
## Quiz (Normal)
고객의 이름, 키, 발사이즈, 선호 브랜드 리스트를 입력받아서 저장하고 출력하는 프로그램을 작성하세요.

argument의 종류는 --name, --height, --foot_size, --wish_list
각각 type은 string, float, int, int
각각 default는 '홍길동', 175.0, 270, [1234]
'''
import os 
import argparse

# parser 정의
parser = argparse.ArgumentParser(description='Argparse Tutorial')
# add_argument()를 통해 argument의 이름, 타입, 기본 값, 도움말을 정의할수 있다.
parser.add_argument('-n','--name', type=str, default='홍길동', help="Score of korean")
parser.add_argument('--height', type=float, default='175.0',help="Score of mathematcis")
parser.add_argument('-f', '--foot_size', type=int, default=270, help = "Score of english")
parser.add_argument('-w', '--wish_list', type=list, default=[1,2,3,4], help = "Score of english")

# add_argment()함수를 호출하면 parser인스턴스 내부에 해당 ㅣㅇ름을 가지는 멤버 변수를 생성
# parse_arg()를 통해 프로그램 실행시 parser가 실행되도록 합니다.
args = parser.parse_args()

# subject_info = {'korean': args.n}
def print_user_info(args):
    print(f"{args.name}의 키는{args.height}이고, 발 사이즈는 {args.wish_list}입니다.")

    

print_user_info(args)
