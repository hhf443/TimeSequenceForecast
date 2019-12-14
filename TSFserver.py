import socket
import os.path
import sys
import threading
import numpy as np
from PIL import Image

import forecast


def main():
    # 创建服务器套接字
    serversocket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    # 获取本地主机名称
    host = socket.gethostname()
    # 设置一个端口
    port = 12345
    # 将套接字与本地主机和端口绑定
    serversocket.bind((host,port))
    # 设置监听最大连接数
    serversocket.listen(5)
    # 获取本地服务器的连接信息
    myaddr = serversocket.getsockname()
    print("服务器地址:%s"%str(myaddr))
    # 循环等待接受客户端信息
    while True:
        # 获取一个客户端连接
        clientsocket,addr = serversocket.accept()
        print("连接地址:%s" % str(addr))
        try:
            t = ServerThreading(clientsocket)#为每一个请求开启一个处理线程
            t.start()
            pass
        except Exception as identifier:
            print(identifier)
            pass
        pass
    serversocket.close()
    pass



class ServerThreading(threading.Thread):
    # words = text2vec.load_lexicon()
    def __init__(self,clientsocket,recvsize=1024*1024,encoding="utf-8"):
        threading.Thread.__init__(self)
        self._socket = clientsocket
        self._recvsize = recvsize
        self._encoding = encoding
        pass

    def run(self):
        print("开启线程.....")


        try:
            #接受数据
            msg = ''
            while True:
                # 读取recvsize个字节
                rec = self._socket.recv(self._recvsize)
                # 解码
                msg += rec.decode(self._encoding)
                # 文本接受是否完毕，因为python socket不能自己判断接收数据是否完毕，
                # 所以需要自定义协议标志数据接受完毕
                if msg.strip().endswith('paramover'):
                    print("调用接收参数方法")
                    msg=msg[:-9]
                    msgList = msg.split('|');
                    print(msgList)
                    specialparams = msgList[0].split()
                    list = msgList[1].split();
                    if not os.path.exists(list[1]):
                        os.rename('fileFromClient.csv', list[1])

                    paramsList = list[1:].copy()

                    if list[0] == 'naive':
                        forecast.naive(paramsList)
                    elif list[0] == 'avg_forecast':
                        forecast.avg_forecast(paramsList)
                    elif list[0] == 'moving_avg_forecast':
                        forecast.moving_avg_forecast(paramsList, specialparams)
                    elif list[0] == 'SES':
                        forecast.SES(paramsList, specialparams)
                    elif list[0] == 'Holt':
                        forecast.Holtmethod(paramsList, specialparams)
                    elif list[0] == 'ARIMA':
                        forecast.ARIMA(paramsList, specialparams)


                    fo = open("result.csv",'rb')
                    while True:
                        filedata = fo.read(1024)
                        if not filedata:
                            break
                        #s.send(filedata)
                        self._socket.send(filedata)
                    fo.close()
                    print("传回数据完毕")
                    break
                elif msg.strip().endswith('fileover'):
                    print("调用接收文件方法")
                    msg = msg[:-8]
                    list = msg.split('|');
                    listdata = list[0:-1]
                    filename = list[-1]

                    fi = open(filename, 'w')   #将str写入文件
                    for items in listdata:
                        fi.write(items)     #读取List的每一行字符串，以,分割表项存入csv
                        fi.write("\n")      #另起一行
                    print("从客户端传来的文件存入完毕")
                    break

            #sendmsg = Image.open(msg)

            # 发送数据
            #self._socket.send(("%s"%sendmsg).encode(self._encoding))
            pass
        except Exception as identifier:
            self._socket.send("500".encode(self._encoding))
            print(identifier)
            pass
        finally:
            self._socket.close()
        print("任务结束.....")

        pass

    def __del__(self):
        pass

if __name__ == "__main__":
    main()