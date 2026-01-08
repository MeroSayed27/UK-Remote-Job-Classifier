import base64
import json
from datetime import datetime

token = 'eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJhdWQiOiJodHRwczovL2lkZW50aXR5dG9vbGtpdC5nb29nbGVhcGlzLmNvbS9nb29nbGUuaWRlbnRpdHkuaWRlbnRpdHl0b29sa2l0LnYxLklkZW50aXR5VG9vbGtpdCIsImlhdCI6MTc2NzgzODc5NywiZXhwIjoxNzY3ODQyMzk3LCJpc3MiOiJmaXJlYmFzZS1hZG1pbnNkay02cjM0eUB0YWJuaW5lLWF1dGgtMzQwMDE1LmlhbS5nc2VydmljZWFjY2NvdW50LmNvbSIsInN1YiI6ImZpcmViYXNlLWFkbWluc2RrLTZyMzR5QHRhYm5pbmUtYXV0aC0zNDAwMTUuaWFtLmdzZXJ2aWNlYWNjb3VudC5jb20iLCJ1aWQiOiJUa3oyRktwWVZERHB2S1dxVTJ6RWRQWjcwTkwyIn0.tHZF1ub1wNfcW9snoqwqQsgsjv-HAjljjYqGbWanME7X-Py5xWbs49sWxvBsifqLp0SjruyUFI3T2ja24c_pATsYkZ57iIfWLtGSAui_WPJg5SiySIq1YWrxqqpvBuVC3agWajOmJrygC3LXM11EgF1Ct4kv69OrjvzU9BHoP_d2mCBqR9DyrxTBEGbMjl8Ng2iOgdbzpCoquXUbzPvv3lHeCh7fpsf3sQ4omYet4XeFyrJaViGaBGS9JBCr5JDW6XjUC9w6QSDN489OldVM8Wg6AKzPK2gpvc_A-is2Dbvcz6xvKAbqbDgdyfNsXIixOc_Ve8uKCG3FXuowe59aEg'

header, payload, sig = token.split('.')

def decode_base64url(s):
    return base64.urlsafe_b64decode(s + '=='[: (4 - len(s) % 4) % 4])

header_data = json.loads(decode_base64url(header))
payload_data = json.loads(decode_base64url(payload))

print('Header:', header_data)
print('Payload:', payload_data)

iat = payload_data['iat']
exp = payload_data['exp']

print('Issued at:', datetime.fromtimestamp(iat))
print('Expires at:', datetime.fromtimestamp(exp))