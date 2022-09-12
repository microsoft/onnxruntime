from http.server import BaseHTTPRequestHandler, HTTPServer
import logging
import time

hostName = "localhost"
serverPort = 80

class MyServer(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(bytes("<html><head><title>https://pythonbasics.org</title></head>", "utf-8"))
        self.wfile.write(bytes("<p>Request: %s</p>" % self.path, "utf-8"))
        self.wfile.write(bytes("<body>", "utf-8"))
        self.wfile.write(bytes("<p>This is an example web server.</p>", "utf-8"))
        self.wfile.write(bytes("</body></html>", "utf-8"))
   
    def do_POST(self):
        request_path = self.path
        '''
        print("\n----- Request Start ----->\n")
        print(request_path)
        
        request_headers = self.headers
        content_length = request_headers.get('content-length')
        length = int(content_length[0]) if content_length else 0
        
        print(request_headers)
        print(self.rfile.read(length))
        print("<----- Request End -----\n")
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        '''
        
        content_length = int(self.headers['Content-Length'])
        content = self.rfile.read(content_length)
        print ("content len: ", content_length)
        print ("content:", content)
        self.send_response(200)
        self.send_header("Content-type", "application/octet-stream")
        self.send_header("Content-length", str(len(content)))
        self.end_headers()
        self.wfile.write(bytes(content))

if __name__ == "__main__":        
    webServer = HTTPServer((hostName, serverPort), MyServer)
    print("Server started http://%s:%s" % (hostName, serverPort))

    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    print("Server stopped.")