import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from tensorflow import keras
from keras import layers

# wczytanie danych
alldata = pd.read_csv("Hotel Reservations.csv", header=0, sep=",", usecols=["Booking_ID", "no_of_adults",
                                                                            "no_of_children",
                                                                            "no_of_weekend_nights",
                                                                            "no_of_week_nights",
                                                                            "type_of_meal_plan",
                                                                            "required_car_parking_space",
                                                                            "room_type_reserved", "lead_time",
                                                                            "arrival_year", "arrival_month",
                                                                            "arrival_date", "market_segment_type",
                                                                            "repeated_guest",
                                                                            "no_of_previous_cancellations",
                                                                            "no_of_previous_bookings_not_canceled",
                                                                            "avg_price_per_room",
                                                                            "no_of_special_requests",
                                                                            "booking_status"])

# wstępne przetworzenie danych
alldata = alldata.drop("Booking_ID", axis=1)

alldata["market_segment_type"] = alldata["market_segment_type"].replace({"Offline": 0, "Online": 1, "Corporate": 2,
                                                                         "Aviation": 3, "Complementary": 4})
alldata["type_of_meal_plan"] = alldata["type_of_meal_plan"].replace({"Not Selected": 0, "Meal Plan 1": 1,
                                                                     "Meal Plan 2": 2, "Meal Plan 3": 3})
alldata["room_type_reserved"] = alldata["room_type_reserved"].replace({"Room_Type 1": 1, "Room_Type 2": 2,
                                                                       "Room_Type 3": 3, "Room_Type 4": 4,
                                                                       "Room_Type 5": 5, "Room_Type 6": 6,
                                                                       "Room_Type 7": 7})
alldata["booking_status"] = alldata["booking_status"].replace({"Not_Canceled": 0, "Canceled": 1})

# wybór różnorodnych cech
Features = ["no_of_adults", "no_of_children", "no_of_weekend_nights", "no_of_week_nights",
            "type_of_meal_plan", "required_car_parking_space", "room_type_reserved", "lead_time",
            "arrival_month", "market_segment_type", "repeated_guest", "avg_price_per_room", "no_of_special_requests"]

# podział na zbiór uczący i testowy
data_train, data_test = train_test_split(alldata, test_size=0.10)

print(len(data_train))
print(len(data_test))

# wyznaczenie x i y dla odpowiednich zbiorów
x_train = pd.DataFrame(data_train[Features])
y_train = pd.DataFrame(data_train["booking_status"])
x_test = pd.DataFrame(data_test[Features])
y_expected = pd.DataFrame(data_test["booking_status"])

# przeskalowanie danych
scaler = StandardScaler()
scaler.fit(x_train)

x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

# regresja logistyczna
model_1 = SGDClassifier(loss="log_loss", penalty="l2")
model_1.fit(x_train_scaled, y_train.values.ravel())
y2_predicted = model_1.predict(x_test_scaled)

print("Regresja logistyczna - ewaluacja")
print(metrics.classification_report(y_expected, y2_predicted, target_names=['Not Canceled', 'Cancelled']))

# Algorytm k-najbliższych sąsiadów
model_2 = KNeighborsClassifier(n_neighbors=13, metric="manhattan")
model_2.fit(x_train_scaled, y_train.values.ravel())
y_predicted = model_2.predict(x_test_scaled)

print("Algorytm k-najbliższych sąsiadów - ewaluacja")
print(metrics.classification_report(y_expected, y_predicted, target_names=['Not Canceled', 'Cancelled']))

# Sieć neuronowa
model_3 = keras.Sequential([
    keras.Input(shape=(None, 13)),
    layers.Dense(512, activation="relu"),
    layers.Dense(256, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(8, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model_3.summary()

model_3.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])

model_3.fit(x_train_scaled, y_train, epochs=100, validation_data=(x_test_scaled, y_expected))

y_predicted = model_3.predict(x_test_scaled)
print("Sieć neuronowa - ewaluacja")
print(metrics.classification_report(y_expected, y_predicted.round(), target_names=['Not Canceled', 'Cancelled']))
