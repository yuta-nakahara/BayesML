<img src="../logos/BayesML_logo.png" width="200">

# マークダウン記法サンプル

<div style="text-align:right;float:right">
作成：中原
</div>

## 見出し

書き方
```
# 見出し1
## 見出し2
### 見出し3
```

出力
> # 見出し1
> ## 見出し2
> ### 見出し3

## 箇条書き

書き方
```
* 項目
* 項目
    * 項目
1. aaa
1. bbb
    1. ccc
```

出力
> * 項目
> * 項目
>     * 項目
> 1. aaa
> 1. bbb
>     1. ccc

## 引用

書き方
```
> 引用文
```

出力
> > 引用文

## コード

書き方
```
    ``` python
    import numpy as np
    ```
    インラインで書くには`import numpy as np`などとする．
```

出力
> ``` python
> import numpy as np
> ```
> インラインで書くには`import numpy as np`などとする．

## 表

書き方
```
|左揃え|右揃え|中央揃え|
| :--- | ---: | :---: |
| c    | ddd  | 222   |
| eee  | fff  | 33    |
```

出力
> |左揃え|右揃え|中央揃え|
> | :--- | ---: | :---: |
> | c    | ddd  | 222   |
> | eee  | fff  | 33    |

## ハイパーリンク

書き方
```
[Google](https://www.google.com/)
```

出力
> [Google](https://www.google.com/)

## 画像の挿入

書き方

基本
```
![logo](../logos/BayesML_logo.png)
```

大きさを調整したいとき
```
<img src="../logos/BayesML_logo.png" width="200">
```

相対パスについては[こちら](https://webliker.info/78726/)を参照

出力

> ![logo](../logos/BayesML_logo.png)

> <img src="../logos/BayesML_logo.png" width="200">

## 数式

書き方
```
$\frac{1}{\sqrt{2 \pi \sigma^2}} \exp \left( -\frac{1}{2 \sigma^2}(x-\mu)^2 \right)$ 
$$\frac{1}{\sqrt{2 \pi \sigma^2}} \exp \left( -\frac{1}{2 \sigma^2}(x-\mu)^2 \right)$$
```

出力
> $\frac{1}{\sqrt{2 \pi \sigma^2}} \exp \left( -\frac{1}{2 \sigma^2}(x-\mu)^2 \right)$
> $$\frac{1}{\sqrt{2 \pi \sigma^2}} \exp \left( -\frac{1}{2 \sigma^2}(x-\mu)^2 \right)$$

## 装飾

書き方
```
**太字**
*斜体*
~~打消し線~~
```
ショートカットはそれぞれ，Ctrl+B, Ctrl+I, Alt+S.

出力
> **太字**
> *斜体*
> ~~打消し線~~

## 課題

このpdfを出力するためのmdファイルを作成せよ．
