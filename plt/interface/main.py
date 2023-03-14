from plt.dl_logic.model import initialize_model, load_weights, prediction, select_one_text


if __name__ == "__main__":

    model = initialize_model()
    load_weights(model)
    X = select_one_text()
    predictions = prediction(model, X)
    print(predictions)
