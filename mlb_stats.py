import base64
import requests


url = 'https://api.mysportsfeeds.com/v1.2/pull/mlb/2017-regular/cumulative_player_stats.json'
username = 'jsh2201'
password = 'dl4cv'

def get_slg(last, first):
	global url, username, password
	params = {'url': url, 'username': username, 'password': password}
	slg = send_request(params, last, first)

	return slg

def send_request(params, last, first):
	try:
		response = requests.get(
			url=url,
			params={
				"player": "{}-{}".format(first, last),
				"playerstats": "SLG"
			},
			headers={
				"Authorization": "Basic " + base64.b64encode('{}:{}'.format(params['username'], params['password']).encode('utf-8')).decode('ascii')
			}
		)

		# print('Response HTTP Status Code: {status_code}'.format(
		    # status_code=response.status_code))
		# print('Response HTTP Response Body: {content}'.format(
		#     content=response.content))


		# Return relevant part of response
		res = response.json()
		return res['cumulativeplayerstats']['playerstatsentry'][0]['stats']['BatterSluggingPct']['#text']

	except requests.exceptions.RequestException as err:
		print('HTTP Request failed')



# Test
slg = get_slg('abreu', 'jose')
print(slg)