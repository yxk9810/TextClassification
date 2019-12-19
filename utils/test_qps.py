#coding: utf-8
import socket
import qa_service_pb2
import struct
import time
import json 
import sys
import numpy as np

def socket_read_n(sock, n):
    buf = b''
    while n > 0:
        data = sock.recv(n)
        if data == b'':
            raise RuntimeError('Unexpected connection close')
        buf += data
        n -= len(data)
    return buf

class Client:
    def __init__(self, ip_address, port, timeout=100):
        self.address = (ip_address, port)
        self.timeout = float(timeout) / 1000

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.settimeout(self.timeout)

    def connect(self):
        try:
            self.socket.connect(self.address)
            return True
        except socket.error:
            return False

    def get_answer(self,line):
        import json
        query_segs = [w.split('#pos#')[0] for w in line.strip().split(' ')]
        start = time.time()  # query_segs = ['王源','抽烟','了','吗','?']
        try:
            query_pos = [int(w.split('#pos#')[1]) for w in
                         line.strip().split(' ')]  # word_token = qa_service_pb2.WordInfo()
        except Exception as e:  # new_request = qa_service_pb2.IntentServiceRequest()
            pass
        new_request = qa_service_pb2.IntentServiceRequest()
        new_request.request_id = 0
        for idx, w in enumerate(query_segs):  # new_request.request_id = 2
            word_token = qa_service_pb2.WordInfo()
            word_token.token = w.encode('utf-8')
            word_token.pos_id = query_pos[idx]  # w.encode('utf-8')
            new_request.query_segs.extend([word_token])  # query_segs)
        #
        protobuf_message = new_request.SerializeToString()
        header = struct.pack('II', len(protobuf_message), 4)
        #
        try:
            self.socket.sendall(header + protobuf_message)
        except Exception as e:
            print(e)
        try:
            len_buf = socket_read_n(self.socket, 8)
            header_len = struct.unpack('II', len_buf)[0]
            header_flag = struct.unpack('II', len_buf)[1]
            message_buf = socket_read_n(self.socket, header_len)
            #
            message = qa_service_pb2.IntentServiceResponse()
            message.ParseFromString(message_buf)
            label = np.argmax(message.probs)  # ParseFromString(message_buf)
            # print(message.probs)  # ParseFromString(message_buf)
            # print('message .....')  # ParseFromString(message_buf)
            # print(time.time() - start)  # ParseFromString(message_buf)
        #     return message.answer
        except Exception as e:
            print(e)
            # print(e)  # print(time.time()-start)#ParseFromString(message_buf)
            # print(line)  # print(time.time()-start)#ParseFromString(message_buf)
            # sys.exit(1)  # print(time.time()-start)#ParseFromString(message_buf)
            # print("error")
            # return ""
        #print('cost request'+str(time.time()-start))#query_pos = [16,31,31,36,34]
        return 'get answer '#     return ""

    def __del__(self):
        self.socket.close()
thread_num =1000
total_cost = [0]*thread_num
def execute_request(data,i):
    time.sleep(i*0.002)#t.connect()# if not client.connect():
    start = time.time() #time.sleep(i*0.05)#t.connect()# if not client.connect():
    t = Client('127.0.0.1', 28040, 10000)
    t.connect()# if not client.connect():
    global total_cost #+= time.clock()-start #time.sleep(i*0.05)#t.connect()# if not client.connect():
    #     print("Connection failed")
    t.get_answer(data)
    total_cost[i]= time.time()-start #time.sleep(i*0.05)#t.connect()# if not client.connect():
lines = [json.loads(line.strip()) for line in open('dev_final_rank30.json','r',encoding='utf-8').readlines()]
threads = [] 
import threading 
for i in range(thread_num):
    t = threading.Thread(target=execute_request,args=[lines[i],i])
    threads.append(t)
start = time.clock() #print(client.get_answer())
if __name__ == "__main__":

    for t in threads: 
        t.start()
    for t in threads:
        t.join()
end = time.clock()
cost = (end-start)/thread_num 
print(" cost ...."+str(cost)) #print(client.get_answer())
print(" total cost ...."+str(sum(total_cost)/thread_num)) #print(client.get_answer())
