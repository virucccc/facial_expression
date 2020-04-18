import sys

from utils import threadHttp as th
from utils import cameraHandler as ch


def main():
    cam = sys.argv[1]
    ip = sys.argv[2]
    port = sys.argv[3]
    server = th.ThreadedHTTPServer(cam, (ip, int(port)), ch.CamHandler)
    print('Сервер запущен http://' + ip + ':' + port + '/cam.mjpg')
    server.serve_forever()


if __name__ == '__main__':
    main()
