# How to Deploy a Keras Model to Production

## Dependencies

```pip install -r requirements.txt -i "https://pypi.douban.com/simple"```

## Usage

Once dependencies are installed, just run this to see it in your browser. 

```python app.py```

That's it! It's serving a saved Keras model to you via Flask. 

Or deploy by gunicorn.

```sh start.sh```

## Examples

Replace the image url as you wish

`curl -d '{"img":"http://pic1.ymatou.com/G02/shangou/M01/EA/27/CgvUBFjk5y-AN3vcAAFy-8IpYrs150_100_133_n_w_o.png"}' -H "Content-type:application/json" modelapi.xiabb.me/predict`

Response is shown below

```
{
  "pred": {
    "mean": 5.367191106561222,
    "std": 1.5243325473321467
  },
  "success": true
}
```

## Reference paper
[https://arxiv.org/abs/1709.05424]
