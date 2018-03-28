from bs4 import BeautifulSoup
import urllib.request
from time import sleep
from datetime import datetime
import pandas as pd

def getWeather():
	#time temperature, wind and pressure
	weather = []
	url = "http://www.eldersweather.com.au/vic/melbourne/melbourne"
	req = urllib.request.urlopen(url)
	page = req.read()
	scraping = BeautifulSoup(page,'lxml')
	#time
	weather.append(scraping.findAll("div",attrs={"class":"update"})[0].text)
	#tempreture
	tmp = scraping.findAll("div",attrs={"class":"now"})[0]
	weather.append(tmp.findAll("div",attrs={"class":"temp"})[0].text)
	#humidity
	tmp = scraping.findAll("div",attrs={"class":"obs"})[0]
	ttmp = tmp.findAll("div",attrs={"class":"attribute"})
	for itm in ttmp:
		if itm.text.find("Humidity") != -1:
			#humidity
			weather.append(itm.findAll("div",attrs={"class":"value"})[0].text)
		if itm.text.find("Pressure") != -1:
			#pressure
			weather.append(itm.findAll("div",attrs={"class":"value"})[0].text)
	return weather

if __name__ == '__main__':
	print("Collecting 10 Weather Data in a Interval of 1 Hour")
	idx = 0
	weatherdata = {'time':[],'tempreture':[],'humidity':[],'pressure':[]}
	while idx < 20:
		print('collecting weather data '+str(idx))
		tmp = getWeather()
		weatherdata['time'].append(tmp[0])
		weatherdata['tempreture'].append(tmp[1])
		weatherdata['humidity'].append(tmp[2])
		weatherdata['pressure'].append(tmp[3])
		idx += 1
		sleep(3600)
	data = pd.DataFrame(weatherdata)
	data.to_csv(r'c:\temp\weather_interval_b.csv')
	print('Endo of Test')