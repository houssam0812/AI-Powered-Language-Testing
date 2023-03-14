from plt.dl_logic.model import initialize_model, load_weights, prediction, select_one_text


if __name__ == "__main__":

    model_1 = initialize_model()
    print("✅ model initialized")
    print("✅ model compiled")
    model_2 = load_weights(model_1)
    print("✅ model weigths loaded")
    X = select_one_text()
    predictions = prediction(model_2, X)
    print(predictions)