'''
PythonでLINE Notifyへ通知を送る
https://qiita.com/akeome/items/e1e0fecf2e754436afc8
'''

import requests


def send_line_notify(notification_message, token_name='ubuntu activity'):
    """
    LINEに通知する
    """
    if token_name == 'ubuntu activity':
        line_notify_token = 'xxxxxx'  # send from 'ubuntu activity' to 'ubuntu activity'
    if token_name == 'ubuntu important':
        line_notify_token = 'yyyyyy'  # send from 'ubuntu important' to 'ubuntu important'

    line_notify_api = 'https://notify-api.line.me/api/notify'
    # headers = {'Authorization': f'Bearer {line_notify_token}'}
    headers = {'Authorization': 'Bearer {}'.format(line_notify_token)}
    # data = {'message': f'message: {notification_message}'}
    data = {'message': 'message: {}'.format(notification_message)}
    requests.post(line_notify_api, headers=headers, data=data)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='''LINE notify''')
    # required argument
    parser.add_argument('--message', '-m', help='a message which you want to notify')
    parser.add_argument('--token', '-t', help='a token group where you want to notify')

    args = parser.parse_args()

    if args.message:
        if args.token:
            send_line_notify(args.message, args.token)
        else:
            send_line_notify(args.message)
    else:
        send_line_notify('ran LINEnotify.py')
