from bs4 import BeautifulSoup
import urllib.request
from time import sleep
from datetime import datetime
import pandas as pd

def getWeather():
	#time temperature, humidity and pressure
	weather = []
	url = "http://m.weatherzone.com.au/vic/melbourne/melbourne"
	req = urllib.request.urlopen(url)
	page = req.read()
	scraping = BeautifulSoup(page,'lxml')
	#time
	weather.append(scraping.findAll("div",attrs={"class":"wrapper"})[1].text)
	#tempreture
	temptmp = scraping.findAll("div",attrs={"id":"today_now"})[0]
	weather.append(temptmp.findAll('span',attrs={"class":"temp temp_15"})[0].text)
	#humidity
	weather.append(scraping.findAll("span",attrs={"class":"rh rh_75"})[0].text)
	#pressure
	prestmp = scraping.findAll("tr",attrs={"class":"odd"})
	pressure = ''
	for itm in prestmp:
		if itm.text.find("Pressure") == 0:
			weather.append(itm.text.split(' ')[1] + ' ' + itm.text.split(' ')[2])
	return weather

if __name__ == '__main__':
	print("Collecting 10 Weather Data by Tempreture Change of more than 1°C")
	idx = 0
	weatherdata = {'time':[],'tempreture':[],'humidity':[],'pressure':[]}
	while idx < 20:
		tmp = getWeather()
		if idx != 0:
			oldtempreture = float(weatherdata['tempreture'][idx-1].replace('°C', ''))
			currtempreture = float(tmp[1].replace('°C', ''))
			if currtempreture - oldtempreture < 1: continue
		print('collecting weather data '+str(idx))
		weatherdata['time'].append(tmp[0])
		weatherdata['tempreture'].append(tmp[1])
		weatherdata['humidity'].append(tmp[2])
		weatherdata['pressure'].append(tmp[3])
		idx += 1
	data = pd.DataFrame(weatherdata)
	data.to_csv(r'c:\temp\weather_change.csv')
	print('Endo of Test')
