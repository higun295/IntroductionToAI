from durable.lang import *

with ruleset('ssm_application_rules'):
    @when_all(m.role == 'employee')
    def enforce_ssm_installation(c):
        print(f"사원 '{c.m.name}'은 스마트폰에 SSM 어플리케이션을 설치 해야 합니다.")
        assert_fact('ssm_application_rules', {'name': c.m.name, 'action': 'collect_info'})

    @when_all(m.action == 'collect_info')
    def collect_device_info(c):
        print(f"회사에서 사원 '{c.m.name}'의 스마트폰 하드웨어 정보와 전화번호를 수집합니다.")

    @when_all(m.role == 'employee')
    def apply_security_sticker(c):
        if c.m.isCommunicationPossible == 'true':
            print(f"사원 '{c.m.name}'의 스마트폰은 현재 통신이 가능하기 때문에 보안 스티커를 부착할 필요가 없습니다.")
        else:
            print(f"사원 '{c.m.name}'의 스마트폰은 현재 통신이 불가능하기 때문에 보안 스티커를 부착해야 합니다.")

with ruleset('campus_security_rules'):
    @when_all(m.role == 'employee')
    def update_employee_location(c):
        if c.m.taggingLocation == 'main_gate_outside':
            print(f"사원 '{c.m.name}'이(가) 캠퍼스에 들어왔습니다.")
        elif c.m.taggingLocation == 'main_gate_inside':
            print(f"사원 '{c.m.name}'이(가) 캠퍼스에서 나갔습니다.")
        elif c.m.taggingLocation == 'security_gate_outside':
            print(f"사원 '{c.m.name}'이(가) 보안구역에 들어왔습니다.")
        elif c.m.taggingLocation == 'security_gate_inside':
            print(f"사원 '{c.m.name}'이(가) 보안구역에서 나갔습니다.")




print('========SSM 어플리케이션 보안 규정========')
assert_fact('ssm_application_rules', {'role': 'employee', 'name': '신희권', 'isCommunicationPossible': 'false'})
print('========캠퍼스 내 보안 규정========')
assert_fact('campus_security_rules', {'role': 'employee', 'name': '신희권', 'taggingLocation': 'main_gate_outside'})
assert_fact('campus_security_rules', {'role': 'employee', 'name': '신희권', 'taggingLocation': 'security_gate_outside'})