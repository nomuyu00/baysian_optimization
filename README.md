<img src="https://img.shields.io/badge/-Python-3776AB.svg?logo=python&style=flat-square">


## Dropoutを使用したベイズ最適化

二つのクラス（DropoutMixBO_BC_UCB と DropoutMixBO_Bandit）の主な違いは以下の点です：

バンディットアルゴリズム:

DropoutMixBO_BC_UCB: BC-UCB（Bernstein Concentration Upper Confidence Bound）を使用
DropoutMixBO_Bandit: 標準的なUCBアルゴリズムを使用


報酬の計算:

DropoutMixBO_BC_UCB: 報酬の二乗値も保存（self.dim_squared_rewards）
DropoutMixBO_Bandit: 報酬の値のみを保存


アクティブ次元の選択:

DropoutMixBO_BC_UCB: より複雑なUCBスコア計算（分散を考慮）
DropoutMixBO_Bandit: シンプルなUCBスコア計算


最適化プロセス:

DropoutMixBO_BC_UCB: より多くの反復回数と生サンプルを使用
DropoutMixBO_Bandit: より少ない反復回数と生サンプルを使用


新しい点の生成:

DropoutMixBO_BC_UCB: スケーリングされた空間で新しい点を生成し、逆変換を適用
DropoutMixBO_Bandit: 元の空間で直接新しい点を生成


ドロップアウトの実装:

DropoutMixBO_BC_UCB: 非アクティブな次元に対してランダムな既存のデータポイントの値を使用
DropoutMixBO_Bandit: 非アクティブな次元に対してランダムな値を生成
