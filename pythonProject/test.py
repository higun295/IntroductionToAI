from durable.lang import *

with ruleset('testRS'):
    @when_all((m.subject == 'World') & (m.body == 'Hello'))
    def say_hello(c):
        print('Hello {0}'.format(c.m.subject))

result = post('testRS', {'subject': 'World', 'body': 'Hello'})
print(result)
