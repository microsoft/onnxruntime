#####################
#### DO NOT EDIT ####
#####################

"""
Owner: isst

This file will start an HTTP server

Run following command
curl http://localhost:8888 --data <the post content>
"""
import os
import platform
import threading
import tornado.ioloop
import tornado.web
import utils
from model import ModelImp

model_imp = None
class MainHandler(tornado.web.RequestHandler):
    def __init__(self, application, request, **kwargs):
        super(MainHandler, self).__init__(application, request, **kwargs)

    def is_binary_content(self):
        content_type = self.request.headers.get("Content-Type", "text/plain")
        return (content_type == "application/binary")

    def post(self):
        if (not self.request.body):
            self.set_status(500)
            self.write("empty request")
            self.finish()
            return

        try:
            if self.is_binary_content():
                response = model_imp.EvalBinary(self.request.body)
                self.set_header("Content-Type", "application/binary")
                self.write(response)
            else:
                response = model_imp.Eval(self.request.body.decode("utf-8"))
                self.set_header("Content-Type", "text/plain")
                self.write(response)
      
            self.finish()
            return
        except Exception as e:
            self.set_status(500)
            self.write("internal server error: %s" % e)
            self.finish()
    
def make_app():
    return tornado.web.Application([(r"/", MainHandler)])

def start(model):
    global model_imp
    model_imp = model
    app = make_app()
    listeningPort = utils.get_listening_port(8888)
    app.listen(listeningPort)
    print("Will listen on port "+str(listeningPort)+".   To invoke manually, run: curl http://localhost:"+str(listeningPort)+" --data <the post content>")
    print("running \n")
    tornado.ioloop.IOLoop.current().start()

if __name__ == "__main__":
    model = ModelImp()
    start(model)
