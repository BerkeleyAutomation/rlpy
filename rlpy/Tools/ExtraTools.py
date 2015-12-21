from datetime import datetime

def getTimeStr():
	return datetime.now().strftime('%Y%m%d_%H_%M')