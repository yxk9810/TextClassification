#coding: utf-8
import socket
import qa_service_pb2
import struct
import time
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

    def get_answer(self):
        writer = open('result_0812.txt','a+',encoding='utf-8')
        i =0 #writer = open('result_0524.csv','a+',encoding='utf-8')
        with open('seg_data_sample_1w.txt','r',encoding='utf-8') as lines:
            for line in lines:
                query_segs = [w.split('#pos#')[0] for w in line.strip().split(' ')]
                i+=1#query_segs = [w.split('#pos#')[0] for w in line.strip().split(' ')]
                start = time.time()#query_segs = ['王源','抽烟','了','吗','?']
                try: 
                    query_pos = [int(w.split('#pos#')[1]) for w in line.strip().split(' ')]#word_token = qa_service_pb2.WordInfo()
                except Exception as e: #new_request = qa_service_pb2.IntentServiceRequest()
                    continue #print('.....'+str(i))#word_token = qa_service_pb2.WordInfo()
                new_request = qa_service_pb2.IntentServiceRequest()
                new_request.request_id = i
                for idx,w in enumerate(query_segs): #new_request.request_id = 2
                    word_token = qa_service_pb2.WordInfo()
                    word_token.token = w.encode('utf-8')
                    word_token.pos_id = query_pos[idx]#w.encode('utf-8')
                    new_request.query_segs.extend([word_token])#query_segs)
            #
                protobuf_message = new_request.SerializeToString()
                header = struct.pack('II',len(protobuf_message), 4)
                #
                try:
                    self.socket.sendall(header + protobuf_message)
                except Exception as e:
                     print(e)
                try:
                    len_buf = socket_read_n(self.socket, 8)
                    header_len = struct.unpack('II', len_buf)[0]
                    print(header_len)# = struct.unpack('II', len_buf)[0]
                    header_flag = struct.unpack('II', len_buf)[1]
                    message_buf = socket_read_n(self.socket, header_len)
                    print(message_buf)# = qa_service_pb2.IntentServiceResponse()
                #
                    message = qa_service_pb2.IntentServiceResponse()
                    message.ParseFromString(message_buf)
                    label = np.argmax(message.probs)#ParseFromString(message_buf)
                    print(message.probs)#ParseFromString(message_buf)
                    print('message .....')#ParseFromString(message_buf)
                    writer.write(''.join([w.split('#pos#')[0] for w in line.strip().split(' ')])+'\t'+str(label)+'\t'+''.join([str(v) for v in message.probs])+'\n')
                    print(time.time()-start)#ParseFromString(message_buf)
                #     return message.answer
                except Exception as e:
                    print(e)#print(time.time()-start)#ParseFromString(message_buf)
                    print(line)#print(time.time()-start)#ParseFromString(message_buf)
                    sys.exit(1)#print(time.time()-start)#ParseFromString(message_buf)
                     # print("error")
                     # return ""

        writer.close()
        return 'get answer '#     return ""

    def __del__(self):
        self.socket.close()

