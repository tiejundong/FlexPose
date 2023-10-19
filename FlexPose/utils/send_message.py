import os
import requests


class SendWechat:
    def __init__(self, args):
        self.use_wc = args.use_wc
        self.token = args.wc_token
        self.template = 'html'
        self.timeout = 1

        self.proxies = {
            'http': os.environ.get('http_proxy'),
            'https': os.environ.get('https_proxy'),
        }

        self.cwd = os.path.split(os.getcwd())[-1]

    def __call__(self, content):
        if self.token is not None and self.use_wc:
            try:
                self.send(content)
            except:
                print('Fail in sending message')

    def send(self, meg):
        title = f'Dir: {self.cwd}'
        content = meg
        url = f'http://www.pushplus.plus/send?token={self.token}&title={title}&content={content}&template={self.template}'
        r = requests.get(url=url, proxies=self.proxies, verify=False, timeout=self.timeout)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_wc', type=str, default=True, help='send status')
    parser.add_argument('--wc_token', type=str, default=None, help='')
    args = parser.parse_args()

    test = SendWechat(args)
    test('testing')

    print('DONE')




























