import json,urllib.request

# Sign in
url='http://127.0.0.1:8080/api/auth/signin'
creds={'username':'farmer1','password':'farmer123'}
req=urllib.request.Request(url, data=json.dumps(creds).encode('utf-8'), headers={'Content-Type':'application/json'})
try:
    resp=urllib.request.urlopen(req, timeout=10)
    body=json.loads(resp.read().decode())
    jwt=body.get('token') or body.get('jwt') or body.get('accessToken') or body.get('token')
    if not jwt:
        jwt = body.get('token') or body.get('jwt')
    print('SIGNIN', body)
except Exception as e:
    print('SIGNIN ERROR', e)
    jwt=None

# Call cropRecommendation
if jwt:
    url='http://127.0.0.1:8080/api/cropRecommendation'
    data={
      'state':'Karnataka','district':'Mandya','season':'Kharif','soilType':'Clay','nitrogen':80,'phosphorus':40,'potassium':120,'temperature':28,'humidity':70,'rainfall':180
    }
    headers={'Content-Type':'application/json','Authorization':f'Bearer {jwt}'}
    req=urllib.request.Request(url, data=json.dumps(data).encode('utf-8'), headers=headers)
    try:
        resp=urllib.request.urlopen(req, timeout=10)
        print('CROP RESPONSE', resp.read().decode())
    except Exception as e:
        print('CROP ERROR', e)
else:
    print('No JWT, skipping crop call')
