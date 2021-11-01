# classification

streamlitではsvgファイルを表示できなかったため、dtreezvizが直接使えない。。よって、事前にGoogle Colaboratoryで学習し、svgファイルを取得。pngファイルに書き換え、本教材で使用した。

Google Colaboratoryのコードは以下の通り。

```
def my_dtreevis(feature1,feature2 ):
  X = df[[feature1, feature2]]
  y = df.Species

  y = y.replace("Iris-setosa", 0)
  y = y.replace("Iris-versicolor", 1)
  y = y.replace("Iris-virginica", 2)

  # データの取得
  # 学習データとテストデータに分割
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)
  # train_X, test_X, train_y, test_y = load_ML_data(feature1, feature2, train_num = 600)

  # 木の深さを3に制限
  clf = DecisionTreeClassifier(random_state=0, max_depth=3)
  # 学習
  clf = clf.fit(X_train, y_train)

  viz = dtreeviz(clf, X_train, y_train,
                  target_name='Species',
                  feature_names=[feature1, feature2],
                  class_names=['setosa', 'versicolor','virginica']) 
  
  fname = "/path/to/your/save_folder"
  viz.save(fname)
```
