from ml import fit, LogisticModel
def test_fit():
    x = [0,1]
    y =[0,1]
    slope, intercept = fit(x,y)
    assert slope==9.812037170985635
    assert intercept==5.175316482607568

def test_probabilities():
    lm = LogisticModel()
    x = [0,0.2,0.4,0.6,0.8,1]
    y = [0,0.2,0.4,0.6,0.8,1]
    lm.fit(x,y)
    ps = lm.predit_proba(x)
    for p in ps:
        asser p>=0
        asser p<=1