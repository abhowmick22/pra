package edu.cmu.ml.rtw.pra.mallet_svm.kernel;

import edu.cmu.ml.rtw.pra.mallet_svm.common.SparseVector;
import edu.cmu.ml.rtw.pra.mallet_svm.libsvm.svm_node;
import java.io.Serializable;

/**
 *  <code>LinearKernel</code> implements a linear kernel function.
 * @author Syeed Ibn Faiz
 */
public class LinearKernel implements CustomKernel, Serializable {

    @Override
    public double evaluate(svm_node x, svm_node y) {                        
        if (!(x.data instanceof SparseVector) || !(y.data instanceof SparseVector)) {
            throw new RuntimeException("Could not find sparse vectors in svm_nodes");
        }        
        SparseVector v1 = (SparseVector) x.data;
        SparseVector v2 = (SparseVector) y.data;
                        
        return v1.dot(v2);
    }    
}
