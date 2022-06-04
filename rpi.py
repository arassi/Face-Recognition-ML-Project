import paho.mqtt.client as mqtt
import time
import grovepi
from grovepi import *
from grove_rgb_lcd import *

LCD_LINE_LEN = 16
        
def custom_lcd(client, userdata, message):
    setText(str(message.payload, "utf-8"))  
    lcd.setText_norefresh("It is: " + msg.topic + "'s face!" + str(msg.payload, "utf-8"))
    
def on_connect(client, userdata, flags, rc):
    print("Connected to server (i.e., broker) with result code "+str(rc))

    #subscribe to topics of interest here
    client.subscribe("arassi/lcd")
    client.message_callback_add("arassi/lcd", custom_lcd)

#Default message callback. Please use custom callbacks.
def on_message(client, userdata, msg):
    #print("on_message: " + msg.topic + " " + str(msg.payload, "utf-8"))
    lcd.setText_norefresh("It is: " + msg.topic + "'s face!" + str(msg.payload, "utf-8"))

if __name__ == '__main__':
    #this section is covered in publisher_and_subscriber_example.py
    client = mqtt.Client()
    client.on_message = on_message
    client.on_connect = on_connect
    client.connect(host="broker.emqx.io", port=1883, keepalive=60)

    while True:
        time.sleep(1)
