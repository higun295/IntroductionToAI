from durable.lang import *

with ruleset('install_ssm_application'):
    # 사원은 스마트폰에 SSM 어플리케이션을 설치 해야 한다.
    @when_all(m.role == 'employee')
    def enforce_ssm_installation(c):
        print(f"사원 '{c.m.name}'은 스마트폰에 SSM 어플리케이션을 설치 해야 합니다.")
        assert_fact('install_ssm_application', {'name': c.m.name, 'action': 'collect_info'})

    # SSM 어플리케이션을 설치하면 하드웨어 정보와 전화번호를 수집한다.
    @when_all(m.action == 'collect_info')
    def collect_device_info(c):
        print(f"회사에서 사원 '{c.m.name}'의 스마트폰 하드웨어 정보와 전화번호를 수집합니다.")

    # 스마트폰은 반드시 통신이 가능한 상태여야하며, 불가능한 경우에는 보안 스티커를 부착한다.
    @when_all(m.role == 'employee')
    def apply_security_sticker(c):
        if c.m.isCommunicationPossible == 'true':
            print(f"사원 '{c.m.name}'의 스마트폰은 현재 통신이 가능하기 때문에 보안 스티커를 부착할 필요가 없습니다.")
        else:
            print(f"사원 '{c.m.name}'의 스마트폰은 현재 통신이 불가능하기 때문에 보안 스티커를 부착해야 합니다.")

assert_fact('install_ssm_application', {'role': 'employee', 'name': '신희권', 'isCommunicationPossible': 'false'})

