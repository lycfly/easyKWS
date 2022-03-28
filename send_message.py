
import requests
import json
test_acc = 0.98
test_loss = 0.01


def send_message(message):
    data = {
        "token": "b66d25faea30",
        "title": "easyKWS",
        "name": "EasyKWS Training Message",
        "content": message
        }
    resp = requests.post("https://www.autodl.com/api/v1/wechat/message/push",
                        json=data)
    print(resp.content.decode())

if __name__ == "__main__":
    message = "Test accuracy: %.03f, Test loss %f\\n" % (100 * test_acc, test_loss)
    send_message(message)