package moa.classifiers.randomUndersampling;

import com.yahoo.labs.samoa.instances.MultiLabelInstance;
import com.yahoo.labs.samoa.instances.Prediction;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.AbstractMultiLabelLearner;
import moa.core.Measurement;

public class randomUndersampling extends AbstractMultiLabelLearner {
    @Override
    public void trainOnInstanceImpl(MultiLabelInstance instance) {

    }

    @Override
    public Prediction getPredictionForInstance(MultiLabelInstance inst) {
        return null;
    }

    @Override
    public void resetLearningImpl() {

    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return new Measurement[0];
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {

    }

    @Override
    public boolean isRandomizable() {
        return false;
    }
}
