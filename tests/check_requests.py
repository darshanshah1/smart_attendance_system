import requests

status_dict = requests.get(headers={"Authorization": "Bearer YqVVptS1wPHntC6mOwszrG2xHAdsH1Zy6ytgkzLAAhu"},url="http://0.0.0.0:8521/get-current-status")
activity_dict = requests.get(headers={"Authorization": "Bearer YqVVptS1wPHntC6mOwszrG2xHAdsH1Zy6ytgkzLAAhu"},url="http://0.0.0.0:8521/get-activities").json()
print_s = ""
if status_dict.status_code == 200:
	status_dict = status_dict.json()
else:
	print(status_dict.content)
for i in status_dict['data'].keys():
	
	print_s += f"> {i} COUNT = {status_dict['data'][i]['count']}| "
	
	for j in status_dict['data'][i].values():
		if j == status_dict['data'][i]['count']:
			continue
		print_s += f"{', '.join(list(j.keys()))}"
	print_s += '\n'

print_s += f"ACTIVITIES COUNT = {activity_dict['count']}"
print(print_s)

