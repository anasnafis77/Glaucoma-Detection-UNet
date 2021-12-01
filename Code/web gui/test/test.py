import re
from collections import Counter
from flask import Flask
from flask import render_template
from flask import request

app = Flask(__name__)


def parser(data):
    pattern = r'\d{1, 3}\.\d{1, 3}\.\d{1, 3}\.\d{1, 3}'

    ips = re.findall(pattern, data)
    results = Counter(ips).most_common(10)
    return results

@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        log = request.files['log_file'].read()
        txt = str(log, 'utf8')  # txt str(log.encode('utf-8))
        result = parser(txt)
        return render_template("index.html", ips=result)
        
    else:
        return render_template("index.html")


if __name__ == '__main__':
    app.run(debug=True)