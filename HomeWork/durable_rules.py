"""
자신의 현업에서 특정 분야 문제에 대해 15개 이상의 규칙을 찾아서
Durable_Reuls로 구현하고 실행됨을 보이시오.
코드와 실행 결과를 제출하고, 4장짜리 슬라이드도 함께 제출하기 바랍니다.

SSM(Smartdevice Security Management) - 출입정책적용, 출입정책해제, 보안구역 입문시
사내에서 사원증 태깅과 나의 움직임 및 행동의 연관성

게이트 종류 : 메인 게이트, 보안구역 게이트
사원 상태 : 캠퍼스 밖, 캠퍼스 안, 보안구역 밖, 보안구역 안
SSM 출입정책 상태 : 정책적용중, 출입정책해제

* 규칙 베이스(rule base) : 전체 규칙의 집합을 관리하는 부분
    - 사원은 스마트폰에 SSM 어플리케이션을 설치해야 한다.
    - 스마트폰에 SSM 어플리케이션을 설치하면 단말기 하드웨어 정보와 전화번호를 수집한다.
    - 스마트폰의 데이터 통신이 불가능한 경우에는 보안스티커를 부착한다.
    - 사원이 SSM 어플리케이션을 실행하면 사원의 위치(보안구역, 캠퍼스 바깥/안)를 확인하고, 출입정책 적용상태를 확인한다.

    - 캠퍼스에는 메인 게이트, 보안구역 게이트. 두 종류의 게이트가 존재한다.
    - 사원은 게이트를 통과할 때 사원증을 태깅한다.
    - 캠퍼스 바깥쪽 게이트에서 사원증을 태깅하면 해당 사원은 캠퍼스 안으로 들어온 것이다.
    - 캠퍼스 안쪽 게이트에서 사원증을 태깅하면 해당 사원은 캠퍼스 밖으로 나간 것이다.
    - 사원이 보안구역 바깥쪽 게이트에서 사원증을 태깅하면 해당 사원은 보안구역에 들어온 것이다.
    - 사원이 보안구역 안쪽 게이트에서 사원증을 태깅하면 해당 사원은 보안구역에서 나간 것이다.

    - 사원이 보안구역 안에 있으면 SSM 출입정책의 상태를 '정책적용중'으로 변경한다.
    - SSM의 출입정책 상태가 '정책적용중'으로 변경되면 스마트폰의 모든 카메라 기능을 차단한다.
    - 사원이 보안구역 안에 있고, SSM 출입정책의 상태가 '출입정책해제'이고, 이 상태로 5초가 지나면 푸시 알림을 보낸다.
    - 사원이 보안구역 안에 있고, SSM 출입정책의 상태가 '출입정책해제'이고, 푸시 알림을 받은 기록이 있고, 이 상태로 1분이 지나면 카카오톡과 문자를 전송한다.
    - 사원이 보안구역 안에 있고, SSM 출입정책의 상태가 '출입정책해제'이고, 푸시 알림을 받은 기록이 있고, 카카오톡과 문자를 받은 기록이 있고, 이 상태로 1분이 지나면 SSM 화면에 '이상 패턴. 확인이 필요합니다' 문구를 표시한다.
    - 사원이 보안구역에서 나올 때 SSM 화면에 '이상 패턴. 확인이 필요합니다' 문구가 표시되어 있으면 보안요원이 카메라 촬영 내역 및 삭제 내역을 검사한다.
    - 카메라 촬영 기록 및 삭제 내역이 있으면 보안 위규 처리한다.
    - 카메라 촬영 기록 및 삭제 내역이 없으면 보안 위규 처리하지 않는다.
    - 사원이 보안구역 안에 있으면 사원은 SSM 출입정책은 해제할 수 없다.
    - 사원이 보안구역 밖에 있으면 사원은 SSM 출입정책을 수동해제 할 수 있다.
    - 사원이 보안구역 밖에 있고, 캠퍼스 밖에 있으면 SSM 출입정책을 해제한다.

    - 사원은 캠퍼스 내 촬영을 할 수 없다.
    - 보안 요원은 사원의 스마트폰을 불시에 점검할 수 있는 권한이 있다.
    - 보안 요원이 사원의 스마트폰에서 촬영 기록 및 삭제 내역을 발견하면 보안 위규 처리된다.
"""

from durable.lang import *

with ruleset('test'):
    # antecedent
    @when_all(m.subject == 'World')
    def say_hello(c):
        # consequent
        print('Hello {0}'.format(c.m.subject))

result = post('test', {'subject': 'World'})
print(result)