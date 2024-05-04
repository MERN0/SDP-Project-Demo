import streamlit as st
import pandas as pd
# from sklearn.preprocessing import StandardScaler
import model_training as trained_model
import numpy as np
from sklearn.metrics import confusion_matrix

# Load your trained model (replace with your actual model loading logic)
model = trained_model.model


# Define a function to map predictions to labels (legit/fraud)
def predict_label(prediction):
    if prediction == 0:
        return "Legit"
    else:
        return "Fraud"


# Define the main app function
def main():
    st.title("Fraud Detection App")

    file_uploaded = st.file_uploader("Upload a CSV file containing transactions", type=["csv"])
    if file_uploaded is not None:
        df = pd.read_csv(file_uploaded)

        # Preprocess the data (ensure it's in the same format as the training data)
        X = df.drop('Class', axis=1)

        # Make predictions
        predictions = model.predict(X)

        # Convert predictions to labels (legit/fraud)
        predictions_label = np.vectorize(predict_label)(predictions)

        # # Display the results
        # st.write("Transaction Predictions:")
        # st.dataframe(pd.DataFrame({'Transaction': df.index, 'Predicted Class': predictions_label}))
        #
        # # Visualize the distribution of predictions (corrected code)
        # st.bar_chart(pd.Series(predictions).value_counts())  # Convert to a Series for value_counts

        # Transaction Predictions
        st.write("Transaction Predictions:")
        st.dataframe(pd.DataFrame({'Transaction': df.index, 'Predicted Class': predictions_label}))

        # Confusion Matrix
        st.header("Confusion Matrix")
        y_true = df['Class']  # Assuming 'Class' column exists
        cm = confusion_matrix(y_true, predictions)
        st.write(cm)

        # Transaction Counts
        st.header("Transaction Counts")
        legit_count = len(df[df['Class'] == 0])
        fraud_count = len(df[df['Class'] == 1])
        st.write("Legit Transactions:", legit_count)
        st.write("Fraudulent Transactions:", fraud_count)

        # Visualize predictions distribution
        st.bar_chart(pd.Series(predictions).value_counts())

if __name__ == '__main__':
    main()