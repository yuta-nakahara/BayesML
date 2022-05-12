<img src="../../logos/BayesML_logo.png" width="200">

# Numpy, Scipyに関する補足

<div style="text-align:right;float:right">
作成：中原
</div>

ipythonノートブックのチュートリアル以外のnumpy, scipy利用例については以下を覚えておくとよい．
* numpyのビューとコピーの違いに注意．特に，更新してはいけない値のビューを生成しないようにする．
* 乱数生成はnumpyの[Random Generator](https://numpy.org/doc/stable/reference/random/generator.html)
* 確率分布に対する様々な操作は[scipi.stats](https://docs.scipy.org/doc/scipy/reference/stats.html)
* 以下のような関数も便利
  * [ガンマ関数](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.gamma.html#scipy.special.gamma)
  * [ガンマ関数の対数](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.gammaln.html#scipy.special.gammaln)
  * [ディガンマ関数](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.digamma.html#scipy.special.digamma)
  * [ポリガンマ関数](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.polygamma.html#scipy.special.polygamma)
  * [逆行列](https://numpy.org/doc/stable/reference/generated/numpy.linalg.inv.html#numpy.linalg.inv)
  * [一般化逆行列](https://numpy.org/doc/stable/reference/generated/numpy.linalg.pinv.html#numpy.linalg.pinv)
  * [連立方程式Ax=bの解](https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html#numpy.linalg.solve)（上述の逆行列を計算した後A^{-1}bを計算するより高速かつ高精度）
  * [行列式](https://numpy.org/doc/stable/reference/generated/numpy.linalg.det.html#numpy.linalg.det)
  * [行列式の対数](https://numpy.org/doc/stable/reference/generated/numpy.linalg.slogdet.html#numpy.linalg.slogdet)
  * [ロジスティックシグモイド関数](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.expit.html)
  * [x log y](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.xlogy.html#scipy.special.xlogy)（0 log 0でエラーにならない）
  * [logsumexp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.logsumexp.html#scipy.special.logsumexp)
  * [logaddexp](https://numpy.org/doc/stable/reference/generated/numpy.logaddexp.html)
* その他のnumpyのTips
  * 対角行列との積はnumpy.diagを用いるよりブロードキャストを利用したほうがよい．
  * 積のトレースTr{AB}を計算するときは，np.sum(A * B)が速い．
