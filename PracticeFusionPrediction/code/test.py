import numpy as np
from numpy.testing import assert_equal
from scipy.sparse import csr_matrix
 
import sklearn
 
from sklearn.feature_selection import SelectKBest, chi2
 
# Feature 0 is highly informative for class 1;
# feature 1 is the same everywhere;
# feature 2 is a bit informative for class 2.
X = ([[2, 1, 2],
      [9, 1, 1],
      [6, 1, 2],
      [0, 1, 2]])
y = [0, 1, 2, 2]
 
 
def mkchi2(k):
    """Make k-best chi2 selector"""
    return SelectKBest(chi2, k=k)
 
 
def test_chi2():
    """Test Chi2 feature extraction"""
 
    chi = sklearn.feature_selection.chi2(X, y)
    print chi
 
    chi2 = mkchi2(k=1).fit(X, y)
    chi2 = mkchi2(k=1).fit(X, y)
    print chi2.get_support(indices=True), [0]
    print chi2.transform(X), np.array(X)[:, [0]]
 
    chi2 = mkchi2(k=2).fit(X, y)
    print sorted(chi2.get_support(indices=True)), [0, 2]
 
    Xsp = csr_matrix(X, dtype=np.float)
    chi2 = mkchi2(k=2).fit(Xsp, y)
    print sorted(chi2.get_support(indices=True)), [0, 2]
    Xtrans = chi2.transform(Xsp)
    print Xtrans.shape, [Xsp.shape[0], 2]
 
    # == doesn't work on scipy.sparse matrices
    Xtrans = Xtrans.toarray()
    Xtrans2 = mkchi2(k=2).fit_transform(Xsp, y).toarray()
    assert_equal(Xtrans, Xtrans2)
	
if __name__=="__main__":
    test_chi2()