"""
자신의 현업에서 특정 분야 문제에 대해 15개 이상의 규칙을 찾아서
Durable_Reuls로 구현하고 실행됨을 보이시오.
코드와 실행 결과를 제출하고, 4장짜리 슬라이드도 함께 제출하기 바랍니다.

SSM(Smartdevice Security Management) - 출입정책적용, 출입정책해제, 보안구역 입문시
사내에서 사원증 태깅과 나의 움직임 및 행동의 연관성

게이트 종류 : 메인 게이트, 보안구역 게이트
사원 상태 : 캠퍼스 밖, 캠퍼스 안, 보안구역 밖, 보안구역 안
SSM 출입정책 상태 : 정책적용중, 출입정책해제, 이상패턴

* 규칙 베이스(rule base) : 전체 규칙의 집합을 관리하는 부분
    - 사원은 스마트폰에 SSM 어플리케이션을 설치해야 한다.
    - 스마트폰에 SSM 어플리케이션을 설치하면 단말기 하드웨어 정보와 전화번호를 수집한다.
    - 스마트폰의 데이터 통신이 불가능한 경우에는 보안스티커를 부착한다.

    - 캠퍼스에는 메인 게이트, 보안구역 게이트. 두 종류의 게이트가 존재한다.
    - 사원은 게이트를 통과할 때 사원증을 태깅한다.
    - 캠퍼스 바깥쪽 게이트에서 사원증을 태깅하면 해당 사원은 캠퍼스 안으로 들어온 것이다.
    - 캠퍼스 안쪽 게이트에서 사원증을 태깅하면 해당 사원은 캠퍼스 밖으로 나간 것이다.
    - 사원이 보안구역 바깥쪽 게이트에서 사원증을 태깅하면 해당 사원은 보안구역에 들어온 것이다.
    - 사원이 보안구역 안쪽 게이트에서 사원증을 태깅하면 해당 사원은 보안구역에서 나간 것이다.

    - 사원이 보안구역 안에 있으면 SSM 출입정책의 상태를 '정책적용중'으로 변경한다.
    - SSM의 출입정책 상태가 '정책적용중'으로 변경되면 스마트폰의 모든 카메라 기능을 차단한다.
    - 사원이 보안구역 안에 있고, SSM 출입정책의 상태가 '출입정책해제'이고, 이 상태로 5초가 지나면 푸시 알림을 보낸다.
    - 사원이 보안구역 안에 있고, SSM 출입정책의 상태가 '출입정책해제'이고, 이 상태로 1분이 지나면 카카오톡과 문자를 전송한다.
    - 사원이 보안구역 안에 있고, SSM 출입정책의 상태가 '출입정책해제'이고, 이 상태로 3분이 지나면 SSM 화면에 '이상 패턴. 확인이 필요합니다' 문구를 표시한다.
    - 사원이 보안구역에서 나올 때 SSM 화면에 '이상 패턴. 확인이 필요합니다' 문구가 표시되어 있으면 보안요원이 카메라 촬영 내역 및 삭제 내역을 검사한다.
    - 카메라 촬영 기록 및 삭제 내역이 있으면 보안 위규 처리한다.
    - 카메라 촬영 기록 및 삭제 내역이 없으면 보안 위규 처리하지 않는다.
    - 사원이 보안구역 안에 있으면 사원은 SSM 출입정책은 해제할 수 없다.
    - 사원이 보안구역 밖에 있으면 사원은 SSM 출입정책을 수동해제 할 수 있다.
    - 사원이 보안구역 밖에 있고, 캠퍼스 밖에 있으면 SSM 출입정책을 해제한다.
    - 사원이 SSM 어플리케이션을 실행하면 사원의 위치(보안구역, 캠퍼스 바깥/안)를 확인하고, 출입정책 적용상태를 확인한다.

    - 사원은 캠퍼스 내 촬영을 할 수 없다.
    - 보안 요원은 사원의 스마트폰을 불시에 점검할 수 있는 권한이 있다.
    - 보안 요원이 사원의 스마트폰에서 촬영 기록 및 삭제 내역을 발견하면 보안 위규 처리된다.
"""

from durable.lang import *

with ruleset('ssm_application_rules'):
    @when_all(m.role == 'employee')
    def enforce_ssm_installation(c):
        print(f"사원 '{c.m.name}'은 스마트폰에 SSM 어플리케이션을 설치 해야 합니다.")
        post('ssm_application_rules', {'name': c.m.name, 'action': 'collect_info'})

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
    def track_employee_location(c):
        if c.m.taggingLocation == 'main_gate_outside':
            print(f"사원 '{c.m.name}'이 캠퍼스에 들어왔습니다.")
        elif c.m.taggingLocation == 'main_gate_inside':
            print(f"사원 '{c.m.name}'이 캠퍼스에서 나갔습니다.")
        elif c.m.taggingLocation == 'security_gate_outside':
            print(f"사원 '{c.m.name}'이 보안구역에 들어왔습니다.")
            # print(f"{c.m.name} SSM 출입정책의 상태가 변경됩니다. 현재상태 : '정책적용중'")
            post('campus_security_rules',
                        {'name': c.m.name, 'current_location': 'inside_security_zone', 'ssm_policy_status': 'not_applied', 'duration': 181})
        elif c.m.taggingLocation == 'security_gate_inside':
            print(f"사원 '{c.m.name}'이 보안구역에서 나갔습니다.")
            post('campus_security_rules', {'name': c.m.name, 'ssm_policy_status': 'abnormal_pattern'})

    @when_all((m.current_location == 'inside_security_zone') & (m.ssm_policy_status == 'applied'))
    def blocking_smartphone_camera(c):
        print("SSM 출입정책이 적용되었습니다. 모든 카메라 기능을 차단합니다.")

    @when_all(m.deactivate_button_pressed == 'true')
    def check_deactivate_possible(c):
        if c.m.current_location == 'inside_security_zone':
            print("보안 구역에서는 SSM 출입정책을 해제할 수 없습니다.")
        elif c.m.current_location == 'outside_security_zone' or c.m.current_location == 'inside_campus':
            print("SSM 출입정책을 해제할 수 있습니다.")

    @when_all((m.current_location == 'inside_security_zone') & (m.ssm_policy_status == 'not_applied'))
    def send_push_notification(c):
        if 5 <= c.m.duration < 60:
            print(f"SSM 출입정책이 적용되지 않았습니다. 사원 '{c.m.name}'의 스마트폰에 푸시 알림을 보냅니다.")
        elif 60 <= c.m.duration < 180:
            print(f"SSM 출입정책이 적용되지 않았습니다. 사원 '{c.m.name}'의 스마트폰에 문자 메시지와 카카오톡을 보냅니다.")
        elif c.m.duration >= 180:
            print(f"SSM 출입정책이 적용되지 않았습니다. 사원 '{c.m.name}'의 스마트폰에 '이상 패턴. 확인이 필요합니다.' 문구를 띄웁니다.")

    @when_all(m.ssm_policy_status == 'abnormal_pattern')
    def inspect_smartphone(c):
        print("SSM 이상상태 확인. 보안요원의 점검이 필요합니다.")
        post('photo_security_rules',
                    {'name': c.m.name, 'role': 'security_guard', 'action': 'inspect_smartphone', 'photo_saved': 'true'})


with ruleset('photo_security_rules'):
    @when_all((m.action == 'take_photo') & ((m.current_location == 'inside_campus') | (m.current_location == 'inside_security_zone')))
    def photo_violation(c):
        print(f"사원 '{c.m.name}'이 캠퍼스 내에서 촬영을 시도했습니다.")
        post('photo_security_rules',
                    {'name': c.m.name, 'role': 'security_guard', 'action': 'inspect_smartphone', 'photo_saved': c.m.photo_saved})

    @when_all((m.role == 'security_guard') & (m.action == 'inspect_smartphone'))
    def inspect_smartphone(c):
        print(f"보안 요원이 사원 '{c.m.name}'의 스마트폰을 점검하고 있습니다.")
        if (c.m.role == 'security_guard') & (c.m.photo_saved == 'true'):
            print(f"보안 요원이 사원 '{c.m.name}'의 스마트폰에서 촬영 기록을 확인했습니다. 보안 위규 처리 되었습니다.")
        else:
            print(f"보안 요원이 사원 '{c.m.name}'의 스마트폰에서 촬영 기록을 확인하지 못했습니다.")


print('========SSM 어플리케이션 보안 규정========')
post('ssm_application_rules', {'role': 'employee', 'name': '신희권', 'isCommunicationPossible': 'false'})
print('========캠퍼스 내 보안 규정========')
post('campus_security_rules', {'role': 'employee', 'name': '신희권', 'taggingLocation': 'main_gate_outside'})
post('campus_security_rules', {'role': 'employee', 'name': '신희권', 'taggingLocation': 'security_gate_outside'})
post('campus_security_rules', {'deactivate_button_pressed': 'true', 'current_location': 'inside_security_zone'})
post('campus_security_rules', {'role': 'employee', 'name': '신희권', 'taggingLocation': 'security_gate_inside'})
print('========캠퍼스 내 사진 촬영 관련 규정========')
post('photo_security_rules', {'name': '신희권', 'action': 'take_photo', 'current_location': 'inside_campus', 'photo_saved': 'true'})
