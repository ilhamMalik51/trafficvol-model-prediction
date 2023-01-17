# Laporan Proyek Machine Learning - Muhammad Ilham Malik

## Domain Proyek
Traffic Volume, Predictive Analysis, Sosial 

Perhitungan lalu lintas adalah perhitungan baik lalu lintas kendaraan atau pejalan kaki, dimana perhitungan ini dilaksanakan sepanjang jalan atau persimpangan tertentu. Perhitungan lalu lintas menyediakan data yang dapat digunakan untuk menghitung lalu lintas harian rata-rata yang dimana menjadi indikator umum sebagai representasi Traffic Volume. Tingginya Traffic Volume dapat menyebabkan berbagai masalah, salah satunya adalah kesempatan dan waktu yang terbuang percuma serta peningkatan stress terhadap kehidupan manusia[1],[2]. Jika dibiarkan, tidak hanya masalah ini dapat mempengaruhi produktivitas individu, bahkan dapat mempengaruhi persaingan kompetitif sebuah negara[3].

Permasalahan ini perlu diselesaikan dengan tujuan agar meningkatkan kualitas hidup manusia dan meningkatkan efisiensi waktu yang digunakan selama perjalanan. Hal ini akan berdampak bertambahnya produktivitas suatu wilayah. Jika setiap wilayah dapat mengatasi masalah ini, maka bukan tidak mungkin akan mempengaruhi produktivitas negara secara keseluruhan. Salah satu solusinya adalah dengan memprediksikan Traffic Volume pada keadaan tertentu seperti waktu, dengan begitu seseorang dapat menghindari waktu tertentu untuk melewati jalan tersebut atau mencari alternatif jalan lain. Selain itu, dapat memahami faktor-faktor yang mempengaruhi tingginya traffic volume akan membuat setiap individu dapat mengambil keputusan yang tepat.

## Business Understanding

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- Seorang individu tidak mengetahui pada keadaan apa saja yang menyebabkan Traffic Volume yang tinggi
- Seorang individu tidak dapat memprediksi apakah pada saat keadaan tertentu Traffic Volume sedang tinggi atau rendah

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Seseorang dapat mengetahui keadaan-keadaan yang memiliki korelasi terhadap Traffic Volume yang tinggi
- Seseorang dapat memprediksikan keadaan Traffic Volume saat ini

### Solution statements
- Memberikan insight keadaan-keadaan yang berkorelasi terhadap Traffic Volume melalui visualisasi data.
- Mengajukan algoritma Linear Regression, Decision Tree Regressor dan Random Forest Regressor untuk memprediksi Traffic Volume dimana metrik yang dijadikan evaluasi model adalah Root Mean Squared Error (RMSE).

## Data Understanding
Dataset Metro Interstate Traffic Volume merupakan dataset yang mengukur volume lalu lintas jalan Westbound Interstate-94, serta mengukur
fitur-fitur cuaca dan hari libur dengan tujuan untuk menjawab pengaruh fitur-fitur tersebut terhadap traffic volume.

Untuk detail lebih lanjut mengenai dataset, dapat dilihat pada [Metro Interstate Traffic Volume Data Set](https://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume). 

### Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:
- holiday : merupakan data kategorikal yang menjelaskan hari libur nasional dan regional US.
- temp : Merupakan data numerik yang menunjukan suhu dalam skala kelvin.
- rain_1h : Merupakan data numerik untuk mengukur hujan yang menggunakan skala milimeter per jam.
- snow_1h : Merupakan data numerik untuk mengukur salju yang menggunakan skala milimeter per jam.
- clouds_all : Merupakan data numerik presentasi awan yang menutupi jalan tersebut.
- weather_main : Merupakan data kategorikal yang menjelaskan deskripsi singkat keadaan cuaca pada daerah tersebut.
- weather_description : Merupakan data kategorikal yang menjelaskan deskripsi panjang keadaan cuaca pada daerah tersebut.
- date_time : Merupakan data bertipe date_time yang membagi dataset setiap jam.
- traffic_volume : Merupakan data numerik yang menunjukan volume lalu lintas.

### Exploratory Data Analysis: Deskripsi Data
Pada bagian ini saya memanfaatkan method `head()`, `info()`, dan `describe()`. Untuk `head()` digunakan untuk melihat secara sekilas struktur data yang ada pada dataset. Metode `info()` digunakan untuk memeriksa tipe data pada setiap atribut. Terakhir, metode `describe()` digunakan untuk memeriksa anomali data atau missing value atau dapat juga untuk menemukan outlier.

Berikut adalah merupakan kode yang digunakan untuk melihat 10 data teratas dari dataset. Kode di bawah ini menggunakan `head()` method yang ada pada objek DataFrame.

```
df = pd.read_csv("Metro_Interstate_Traffic_Volume.csv")
df.head(10)
```
![df_head_screenshot](https://github.com/ilhamMalik51/DicodingAppliedML/blob/b6f010b50202bf8bce08dbd8993f2d58fdd0ab22/Proyek-Pertama/assets/df_head_function_ss.JPG)

Berdasarkan hasil keluaran dari method `head()` tersebut bahwa dataset tersebut memiliki 9 atribut. Dari ke-9 atribut tersebut, terdapat 3 atribut yang sepertinya merupakan tipe data string. Namun nilai dari tipe data string tersebut terdapat nilai yang berulang, jika dilihat baris pertama hingga baris kelima memiliki nilai "clouds", dapat diasumsikan bahwa tipe data dari atribut tersebut merupakan kategorikal. Selain itu terdapat atribut date_time dengan tipe data date-time. Dalam bentuknya yang sekarang nilai pada atribut tersebut belum bisa dianalisis lebih lanjut, maka dari itu atribut ini perlu diproses lebih lanjut dengan tujuan agar ditemukan pola lain selain dari kesembilan fitur-fitur yang sudah ada. 

Selain itu terdapat method yang dapat melihat deskripsi dataset, khususnya akan menampilkan seluruh jumlah baris dataset, tipe setiap atribut dataset, dan jumlah nilai _nonnull_. Method tersebut adalah method `info()` yang terdapat pada DataFrame object.

```
df.info()
```
![df_info_screenshot](https://github.com/ilhamMalik51/DicodingAppliedML/blob/0995b60c3e21aba1785bb6967aa2dcf8e059c128/Proyek-Pertama/assets/df_info_function_Ss.JPG)

Berdasar gambar tersebut dapat diperhatikan bahwa setiap fitur memiliki jumlah nilai data _nonnull_ sebanyak 48204 baris. Hal ini berarti pada setiap fitur tidak terdapat _missing value_. Selain itu, seperti yang telah disinggung sebelumnya terdapat tipe data _object_. Tipe data _object_ ini berarti merupakan semua Python's object, karena dataset ini memiliki format CSV maka tipe data tersebut umumnya text/string.

Selanjutnya terdapat method `describe()`. Method ini akan menampilkan ringkasan statistik dari atribut-atribut numerik.

```
df.describe()
```

![df_describe_screenshot](https://github.com/ilhamMalik51/DicodingAppliedML/blob/f50166b307a9448804d369fc2a691adfa6de2ae5/Proyek-Pertama/assets/df_describe_ss.JPG)

Berdasarkan gambar di atas dapat dilihat ringkasan singkat terkait atribut-atribut numerik yang ada pada dataset. Baris _count_, _mean_, _min_, dan _max_ merupakan baris yang menunjukan jumlah baris, rata-rata, nilai minimum, dan nilai maksimum. Terdapat beberapa hal yang menjadi perhatian dari tabel tersebut:

1. Nilai minimum dari atribut **temp** adalah 0, kita ketahui sebelumnya bahwa atribut temperatur tersebut berada pada skala kelvin dimana nilai 0 derajat kelvin (absolute zero) merupakan hal yang tidak mungkin terjadi selama pengamatan jalan perkotaan. Maka dari itu, baris data yang memiliki nilai atribut **temp** 0 derajat akan di-_drop_.
2. Nilai kedua atribut **rain_1h** dan **snow_1h** pada kuartil pertama (_percentile_ 25%), kuartil kedua (_percentile_ 50%), dan kuartil ketiga (_percentile_ 75%) memiliki nilai yang sama yaitu 0. Hal ini berarti bahwa nilai data yang ada di atribut tersebut hampir seluruhnya bernilai 0 atau dengan kata lain nilai bukan 0 pada atribut tersebut jumlahnya sangat sedikit. Oleh karena itu, kedua fitur ini memiliki kemungkinan yang besar untuk tidak digunakan saat pelatihan model Machine Learning.

Setelah memeriksa data yang memiliki tipe data numerik, saatnya memeriksa tipe data bukan numerik. Seperti yang telah diasumsikan sebelumnya, terdapat 3 tipe data kategorikal. Untuk memeriksa data tersebut dapat menggunakan kode berikut.

```
# df["holiday"].value_counts()
df["weather_main"].value_counts()
# df["weather_description"].value_counts()
```

![df_kategorikal data](https://github.com/ilhamMalik51/DicodingAppliedML/blob/5bfa87efe3138b9783397db91d396129edbdb690/Proyek-Pertama/assets/df_kategorikal_data_overview.JPG)

Hasil dari kode tersebut merupakan salah satu contoh keluaran data kategorikal. Jadi dapat dipastikan bahwa ketiga data tersebut merupakan data kategorikal dimana atribut **holiday** memiliki 12 kategori, atribut **weather_main** memiliki 11 kategori, dan **weather_description** memiliki 38 kategori.

### Exploratory Data Analysis: Data Cleaning
Berdasarkan temuan pada bagian sebelumnya bahwa terdapat baris data dimana nilai atribut **temp** adalah 0. Seperti yang telah disinggung pada bagian tersebut, suhu 0 derajat kelvin tidak mungkin terjadi ketika observasi dataset tersebut. Maka dari itu jika diperiksa menggunakan kode di bawah ini maka akan terlihat baris data yang memiliki nilai atribut **temp** 0 derajat Kelvin.

```
df[df["temp"] < 243]
```
![df_screenshot_below_243](https://github.com/ilhamMalik51/DicodingAppliedML/blob/0e02b42be2ba4fdf27201666786ab22482a37be3/Proyek-Pertama/assets/df_below_243.JPG)

Jika diperhatikan dari keluaran kode tersebut, terdapat 10 baris data yang memiliki nilai atribut **temp** bernilai 0 derajat Kelvin. Baris data ini tergolong _incorrect data_ dan data ini akan sedikit memengaruhi analisis dan prediksi model nantinya. Oleh karena itu, data ini dapat dihapus. Penghapusan data ini tidak akan terlalu memengaruhi dataset karena perbandingan jumlah _incorrect data_ dengan jumlah data keseluruhan yang kecil.

Terdapat beberapa cara untuk menghapus data tersebut, pada kasus ini saya mem-_filter_ data dengan menggunakan kode di bawah ini. Setelah itu, data tersebut dapat diperiksa menggunakan method `describe()` seperti yang telah diperlihatkan pada bagian sebelumnya. Dari tabel yang dihasilkan oleh method tersebut, akan terlihat nilai minimum atribut **temp** yang sebelumnya adalah 0 akan berubah menjadi 244.2 derajat Kelvin.

```
## drop temperature yang hanya bernilai 0 kelvin
df.drop(df[df["temp"] < 243].index, inplace=True)
df.describe()
```

![df_describe_after_dropped_row_data](https://github.com/ilhamMalik51/DicodingAppliedML/blob/2d8b9570aff6c5d17e9fbd1fd8edcc2f622fa8e1/Proyek-Pertama/assets/df_dropped_zero_kelvin_data.JPG)

### Exploratory Data Analysis: Feature Engineering
Sebelum dapat menganalisis lebih lanjut dengan visualisasi data, terdapat satu atribut yang perlu dirubah. Atribut **date_time** yang sekarang kurang memberikan _insight_ yang diperlukan. Oleh karena itu, nilai atribut tersebut dapat dilakukan _feature engineering_ untuk memisahkan setiap nilai yang ada pada data tersebut. Nilai-nilai yang akan saya ambil adalah tahun, bulan, minggu, hari dan jam. Dengan memisahkan nilai-nilai tersebut ditemukan pola/_insight_ lain saat visualisasi data nantinya.

Berikut merupakan cara untuk mengambil nilai-nilai tersebut. Pertama, karena tipe data yang ditampilkan masih `object` maka perlu terlebih dahulu dirubah menjadi format `datetime64`. Setelah itu, setiap nilai data dapat diambil nilainya dan disimpan pada kolom atribut baru.

```
df["date_time"] = pd.to_datetime(df["date_time"],
                                 format='%Y-%m-%d %H:%M:%S',
                                 errors='coerce')

df["date_time_year"] = df["date_time"].dt.year
df["date_time_month"] = df["date_time"].dt.month
df["date_time_day"] = df["date_time"].dt.day
df["date_time_hour"] = df["date_time"].dt.hour
```
Setelah kode tersebut dijalankan maka kolom dataset akan bertambah sebanyak 4 kolom ke sebelah kanan. Perubahan ini dapat dilihat menggunakan method `head()` atau method `info()` seperti yang telah dijelaskan pada bagian sebelumnya.

### Exploratory Data Analysis: Univariate Data
#### Data Numerik
Pada bagian ini akan dilakukan analisis menggunakan teknik univariate EDA. Kode di bawah ini akan menampilkan histogram terkait fitur-fitur numerik yang ada pada dataset. Histogram menunjukan jumlah baris data (sumbu Y) jika diberikan rentang nilai yang ada pada sumbu X.

```
df.hist(bins=100, figsize=(20, 15))
plt.show()
```

![df_histogram_show](https://github.com/ilhamMalik51/DicodingAppliedML/blob/68a79f2accfd3e9432b0ae8d90afd40ac017f558/Proyek-Pertama/assets/hist_numeric_data.png)

Berdasarkan histogram di atas dapat diambil beberapa kesimpulan, antara lain:
1. Atribut-atribut numerik yang ditampilkan memiliki skala yang berbeda-beda, hal ini akan mempengaruhi unjuk kerja dari model Machine Learning yang digunakan.
2. Sesuai asumsi yang sudah disinggung sebelumnya, pada nilai-nilai atribut **rain_1h** dan **snow_1h** lebih dari 48000 baris data atau sekitar 99% data terletak pada nilai 0. Oleh karena itu, ada kemungkinan besar atribut ini tidak akan berpengaruh terhadap nilai prediksi model.
3. Terdapat lebih dari 2000 baris data yang memiliki _traffic volume_ kurang dari 1000.

#### Data Kategorikal
Setelah memeriksa visualisasi untuk tipe data numerik, saatnya menganalisis fitur-fitur tipe data kategorikal. Pada tipe data kategorikal ini dapat menggunakan visualisasi data bar chart. Berikut kode beserta hasil visualisasi data tersebut.

#### Fitur Weather Main

```
df["weather_main"].value_counts().plot(kind="bar", title="Weather Main")
```

![weather_main_categorical](https://github.com/ilhamMalik51/DicodingAppliedML/blob/2dde42d5b6cfbe9b3ba00da65d08b0d2c9091774/Proyek-Pertama/assets/bar_chart_weather_main.png)

Berdasarkan gambar tersebut dapat disimpulkan bahwa 50% dari sampel memiliki keadaan cuaca yang berawan dan cerah.

#### Fitur Weather Description

```
df["weather_description"].value_counts().plot(kind="bar", title="Weather Description")
```

![weather_desc_categorical](https://github.com/ilhamMalik51/DicodingAppliedML/blob/2dde42d5b6cfbe9b3ba00da65d08b0d2c9091774/Proyek-Pertama/assets/bar_chart_weather_desc.png)

Berdasarkan gambar tersebut, karena fitur ini merupakan deskripsi lanjutan daripada atribut **weather_main** maka deskripsi sampel terbanyak merupakan "sky is clear". Sesuai dengan visualisasi sebelumnya, pada visualisasi ini cuaca mendung terbagi menjadi tiga kategori seperti "mist", "overcast clouds", "broken clouds", dan "scattered clouds". Jika ketiga kategori tersebut dijumlahkan maka terdapat 50% observasi data yang mendeskripsikan cuaca yang cerah dan mendung.


#### Fitur Holiday

```
df["holiday"].value_counts().plot(kind="bar", title="Holiday")
```

![holiday_categorical](https://github.com/ilhamMalik51/DicodingAppliedML/blob/2dde42d5b6cfbe9b3ba00da65d08b0d2c9091774/Proyek-Pertama/assets/bar_chart_holiday.png)

Berdasarkan gambar di atas, dapat disimpulkan bahwa lebih dari 97% sampel data observasi bukan merupakan hari libur.

### Exploratory Data Analysis: Multivariate Data

#### Data Numerik
Machine learning akan bekerja lebih baik pada fitur-fitur yang memiliki korelasi linear yang kuat Dapat dilihat bahwa nilai korelasi yang ada pada dataset terhadap atribut label. Nilai korelasi linear ini jatuh pada rentang -1 dan 1, dimana semakin mendekati nilai 1 berarti memiliki korelasi linear positif yang kuat dan apabila mendekati -1 memiliki korelasi linear negatif yang kuat.

Berikut kode yang digunakan untuk menampilkan korelasi linear terhadap fitur target **traffic volume**.

```
corr_matrix = df.corr()
corr_matrix["traffic_volume"].sort_values(ascending=False)
```

![linear_correlation_traffic_volume](https://github.com/ilhamMalik51/DicodingAppliedML/blob/c34afcdde7f9197e2b0b11ec08f48bb262b26a03/Proyek-Pertama/assets/linear_correlation_target_feature.JPG)

Berdasarkan gambar di atas, sudah dapat terlihat fitur-fitur yang memiliki hubungan korelasi linear adalah fitur **date_time** dan **temp**. Atribut selain yang disebutkan sebelumnya memiliki nilai linear yang mendekati 0. Hal ini berarti bahwa fitur-fitur ini memiliki hubungan korelasi linear yang sangat lemah.

Selain itu, agar dapat terlihat lebih jelas, fitur-fitur tersebut dapat divisualisasikan terhadap nilai targetnya menggunakan scatter plot. Teknik ini juga dinamakan teknik multivariate EDA. Agar dapat melakukan hal tersebut, dapat menggunakan kode sebagai berikut.

```
numerical_attributes = ["temp", "date_time_hour", "clouds_all", 
                        "traffic_volume"]

scatter_matrix(df[numerical_attributes], figsize=(12, 8))
```

![scatter_plot_correlation](https://github.com/ilhamMalik51/DicodingAppliedML/blob/e40a9e0b0e368336bcc92e43ab5e40b103cdab74/Proyek-Pertama/assets/scatter_plot_correlation.png)

Berdasarkan gambar di atas dapat dibuktikan bahwa **date_time_hour** memiliki korelasi linear yang menengah (dengan nilai koefisien sekitar 0.3). Bahkan jika diperhatikan lebih dalam lagi, pada atribut tersebut terlihat memiliki hubungan non-linear terhadap atribut target **traffic_volume**. Selain itu karena fitur **temp** memiliki korelasi linear yang lemah (nilai korelasi linear 0.1). Namun, jika diperhatikan lebih seksama terdapat sebuah trend positif pada atribut tersebut. 

Beberapa atribut tidak dimasukan ke dalam visualisasi tersebut adalah salah satunya disebabkan oleh nilai korelasi linear yang sangat kecil mendekati 0. Jika divisualisasikan maka atribut-atribut tersebut akan membentuk segi empat pada diagram scatter plot.

#### Data Kategorikal
Pada bagian ini akan menganalisis pengaruh setiap kategori terhadap rata-rata nilai **traffic_volume**. Pada bagian ini beberapa tipe data kategorikal akan dipisahkan agar visualisasi data terlihat lebih jelas.

```
categorical_features = ["holiday", "weather_main", "weather_description"]
```

#### Fitur Holiday
```
col = categorical_features[0]

sns.catplot(x=col, y="traffic_volume", kind="bar",
            dodge=False, height=4, aspect=3,
            data=df, palette="Set3", legend=True)
plt.title("Rata-rata Volume Traffic terhadap - {}".format(col))
```

![visualisasi_data_holiday](https://github.com/ilhamMalik51/DicodingAppliedML/blob/489b4ad2c665e5445483f476d4989d1d7ed85715/Proyek-Pertama/assets/target_bar_chart_holiday.png)

Berdasarkan visualisasi di atas dapat diambil kesimpulan bahwa, hari libur kurang mempengaruhi **traffic_volume**. Hal ini ditunjukan dengan tingginya **traffic_volume** pada kategori "None" atau berarti bukan hari libur. Lalu jumlah data untuk kategori libur terlampau sangat jauh dibandingkan dengan data libur sehingga kategori ini kurang mempengaruhi fitur target.

#### Fitur Weather Main
```
col = categorical_features[1]

sns.catplot(x="traffic_volume", y=col, kind="bar",
            dodge=False, height=6, aspect=3,
            data=df, palette="Set3", legend=True)
plt.title("Rata-rata Volume Traffic terhadap - {}".format(col))
```

![visualisasi_data_weather_main](https://github.com/ilhamMalik51/DicodingAppliedML/blob/489b4ad2c665e5445483f476d4989d1d7ed85715/Proyek-Pertama/assets/target_bar_chart_weather_main.png)

Berdasarkan visualisasi di atas dapat diambil beberapa kesimpulan sebagai berikut:
1. Pada kategori _clouds_ dan _haze_ menimbulkan rata-rata **traffic volume** lebih dari 3500
2. Pada kategori _squall_ memiliki rata-rata **traffic_volume** yang paling rendah yaitu kurang dari 2000
3. Pada kategori _mist_, _thunderstorm_ dan _fog_ memiliki rata-rata **traffic_volume** di rentang 2500 hingga 3000
4. Pada kategori _snow_, _rain_, dan _drizzle_ memiliki rata-rata **traffic_volume** di rentang 3000 dan 3500

Kesimpulan yang dapat diambil dari pernyataan di atas adalah kategori-kategori tersebut cukup mempengaruhi nilai target dengan selisih setiap target cukup besar yaitu dengan **traffic_volume** bernilai 500.

#### Fitur Weather Description
```
col = categorical_features[2]

sns.catplot(x="traffic_volume", y=col, kind="bar",
            dodge=False, height=10, aspect=3,
            data=df, palette="Set3", legend=True)
plt.title("Rata-rata Volume Traffic terhadap - {}".format(col))
```

![visualisasi_data_weather_desc](https://github.com/ilhamMalik51/DicodingAppliedML/blob/489b4ad2c665e5445483f476d4989d1d7ed85715/Proyek-Pertama/assets/target_bar_chart_weather_desc.png)

Berdasarkan visualisasi di atas kesimpulan yang diambil hampir sama seperti visualisasi yang sebelumnya. Hal ini dikarenakan karena atribut ini merupakan deskripsi lanjutan dari atribut **weather_main**. Pada atribut ini kategori yang ada lebih variatif dibandingkan dengan atribut **weather_main**.
1. Jika diperhatikan, kategori yang menyebabkan **traffic_volume** lebih besar dari nilai 4000 adalah _sleet_, _light shower snow_, _shower snow_, _freezing rain_, dan _proximity snow rain_.
2. Selain itu, kategori _squall_ memiliki nilai **traffic_volume** yang mendekati 2000.
3. Sisanya kategori jatuh pada rentang 2000 hingga 3000 dan 3000 hingga 4000.
4. Kategori pada **weather_description** terlalu bervariatif dan jika dilihat lebih dalam terdapat kategori yang sama namun memiliki penamaan yang sedikit berbeda menyebabkan terlalu banyak kategori yang ada.

Kesimpulan yang diambil adalah bahwa kategori-kategori tersebut tidak cukup berpengaruh untuk memprediksikan nilai **traffic_volume**.

## Data Preparation
Pada bagian ini saya akan menjelaskan data preparation

### Data Preparation: Split Data
Pada bagian ini akan menjelaskan split data. Pada kasus ini, saya menggunakan `StratifiedShuffleSplit()`. _Object_ ini akan membagi secara strata untuk fitur kategorikal **weather_main**. Rasio split data yang digunakan adalah 85:15. Hal ini dikarenakan menurut saya data yang digunakan sudah cukup banyak dan ukuran test-set sudah melebihi 5000 instansi. Maka dari itu akan lebih baik jika jumlah data training jadi lebih banyak.

Berikut adalah contoh kode yang digunakan untuk membagi data menjadi _data training_ dan _data testing_. Kode di bawah ini akan menghasilkan _data training_ sebanyak 40964 dan _data testing_ sebanyak 7230.

```
from sklearn.model_selection import StratifiedShuffleSplit

df_reset_index = df.reset_index()

split = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
for train_index, test_index in split.split(df_reset_index, df_reset_index["weather_main"]):
    strat_train_set = df_reset_index.loc[train_index]
    strat_test_set = df_reset_index.loc[test_index]

feature_columns = ["temp", "weather_main", "date_time_hour"]
label_columns = ["traffic_volume"]

X_train = strat_train_set[feature_columns]
y_train = strat_train_set[label_columns]

X_test = strat_test_set[feature_columns]
y_test = strat_test_set[label_columns]
```

### Data Preparation: One Hot Encoding
One Hot Encoding ini merubah data kategorikal menjadi menjadi data numerik. Caranya adalah dengan memberikan nilai 1 pada kategori aslinya dan membiarkan nilai 0 pada kategori lainnya. metode ini digunakan karena data kategori ini tidak memiliki hubungan ordinal dan Machine Learning umumnya tidak memproses tipe data string.

Berikut merupakan contoh kode yang digunakan untuk merubah tipe data kategorikal menjadi tipe data numerik menggunakan `OneHotEncoder()`. Pada bagian ini diperlihatkan hanya salah satu kategori saja karena di akhir bagian ini akan memanfaatkan pipeline untuk bagian _data preparation_ ini.

```
from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()

weather_desc_cat = df[["weather_main"]]
wd_cat_1hot = cat_encoder.fit_transform(weather_desc_cat)

wd_cat_1hot
```

### Data Preparation: Feature Scaling
Pada bagian ini saya menggunakan `MinMaxScaler()`. `MinMaxScaler()` ini merubah data sebagaimana hingga nilai data jatuh pada rentang 0-1. Cara bekerja `MinMaxScaler()` ini adalah seperti yang diekspresikan berikut: <br/>

`X_hat = (X - min) / (max - min)` <br/>

MinMaxScaler digunakan karena fitur data yang akan diterapkan MinMaxScaler tidak memiliki outlier lalu rentang data yang diubah tergolong tidak terlalu besar, sehingga informasi penting tidak akan hilang.

Berikut merupakan contoh penggunaan `MinMaxScaler()`.

```
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

num_columns = ["temp", "date_time_hour"]
numerical_pipeline = Pipeline([
    ("minmax_scaler", MinMaxScaler())
])

df_num = numerical_pipeline.fit_transform(X_train[num_columns])
df_num.shape
```

### Data Preparation: Transformation Pipeline
Pada bagian ini adalah pengaplikasian dari feature scaling dan perubahan kategorikal data memanfaatkan pipeline yang disediakan oleh library Scikit Learn. Hal ini dilakukan supaya lebih mudah mengaplikasikan transformasi terhadap _test data_. Dengan adanya pipeline ini juga dapat memudahkan transformasi fitur data yang akan diproses untuk model yang sudah di-_deploy_ pada cloud.

Berikut merupakan contoh aplikasi kode pipeline.

```
from sklearn.compose import ColumnTransformer

num_columns = ["temp", "date_time_hour"]
cat_columns = ["weather_main"]

transform_pipeline = ColumnTransformer([
    ("numeric", numerical_pipeline, num_columns),
    ("categorical", OneHotEncoder(), cat_columns)
])

X_train_prepared = transform_pipeline.fit_transform(X_train)
y_train = np.reshape(y_train.to_numpy(), y_train.shape[0])
X_test_prepared = transform_pipeline.fit_transform(X_test)
y_test = np.reshape(y_test.to_numpy(), y_test.shape[0])
```

## Modeling
Pada bagian modeling saya bereksperimen dengan tiga buah model yaitu Linear Regression, Decision Tree Regressor, dan Random Forest Regressor. Sebelum dilanjutkan pada tahap _training_ dan evaluasi model, alangkah baik untuk membuat sebuah fungsi untuk kedua hal tersebut. Fungsi ini akan menerima masukan berupa model, _training data_ dan _test data_, lalu akan menghasilkan nilai metrik RMSE untuk setiap himpunan data untuk setiap model.

```
def evaluation(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    scores = cross_val_score(model, X_train, y_train,
                             scoring="neg_root_mean_squared_error",
                             cv=8)
    train_rmse = -scores.mean()
    
    tv_prediction = model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, tv_prediction))
    
    return train_rmse, test_rmse
```

### Linear Regression
Kelebihan dari Linear Regression ini merupakan model yang paling sederhana dibandingkan model lain yang digunakan dalam eksperimen ini, selain itu kelebihan lainnya adalah waktu training yang cepat.
Kekurangan dari model ini adalah karena model ini termasuk yang paling sederhana, maka model ini masih mengalami underfitting terhadap dataset.

Berikut adalah cara menggunakan model Linear Regression. Hasil dari _training_ dan evaluasi dari model ini ditampilkan pada gambar berikut.

```
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
train_rmse, test_rmse = evaluation(lin_reg, X_train_prepared, y_train, X_test_prepared, y_test)
```

![lin_reg_eval](https://github.com/ilhamMalik51/DicodingAppliedML/blob/76d94224f788525c2a61cce0c3eb9d6ae775f350/Proyek-Pertama/assets/lin_reg_eval.JPG)

### Decision Tree Regressor
Kelebihan dari Decision Tree Regressor adalah model ini dapat mempelajari hubungan non-linear.
Kekurangan dari Decision Tree Regressor pada kasus ini adalah karena model ini lebih kompleks daripada linear regression, model ini lebih rentan terkena overfitting terhadap dataset.

```
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
train_rmse, test_rmse = evaluation(tree_reg, X_train_prepared, y_train, X_test_prepared, y_test)
```

![dt_reg_eval](https://github.com/ilhamMalik51/DicodingAppliedML/blob/76d94224f788525c2a61cce0c3eb9d6ae775f350/Proyek-Pertama/assets/dt_reg_eval.JPG)

### Random Forest Regressor
Kelebihan dari Random Forest Regressor adalah karena model ini merupakan Ensemble Machine Learning, maka model ini merupakan yang paling kompleks.
Kekurangan dari Random Forest Regressor adalah model ini memiliki waktu training yang cukup lama dibanding model yang lain, dan masih terdapat overfitting terhadap dataset.

```
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor()
train_rmse, test_rmse = evaluation(forest_reg, X_train_prepared, y_train, X_test_prepared, y_test)
```

![rf_reg_eval](https://github.com/ilhamMalik51/DicodingAppliedML/blob/76d94224f788525c2a61cce0c3eb9d6ae775f350/Proyek-Pertama/assets/rf_reg_eval.JPG)

### Fine Tuning Random Forest Regressor
Pada bagian ini akan dijelaskan upaya untuk mencari _hyperparameter_ yang optimal dengan tujuan untuk meningkatkan unjuk kerja dari model. Model yang digunakan pada bagian ini adalah Random Forest Regressor karena model ini memiliki unjuk kerja yang terbaik dibandingkan dengan kedua model sebelumnya. _Fine-tune_ model kali ini menggunakan metode _Grid Search_. Berikut adalah kode yang digunakan untuk memulai pencarian.

```
from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators': [3, 10, 30, 50, 70, 100], 'max_features': [2, 4, 6]},
    {'bootstrap': [False], 'n_estimators': [3, 10, 30, 50], 'max_features': [2, 3, 4]},
]

forest_reg_gs = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg_gs, param_grid, cv=5,
                           scoring='neg_root_mean_squared_error',
                           return_train_score=True,
                           verbose=2,)

grid_search.fit(X_train_prepared, y_train)
```

Ketika kode di atas dijalankan pencarian akan dimulai dan akan memakan waktu yang cukup lama. Pertama saya mendeklarasikan _hyperparameter_ yang akan dicari. Lalu menginstansiasi objek model Random Forest Regressor. Lalu memulai pencarian dengan menggunakan method `GridSearchCV()` yang berasal dari library Scikit Learn. Setelah pencarian selesai parameter terbaik dapat diperlihatkan dengan menggunakan kode berikut.

```
grid_search.best_params_
```
Ketika kode di atas dijalankan maka akan ditampilkan _hyperparameter_ terbaik.  _Hyperparameter_ terbaik yang saya dapatkan adalah
- max_features: 6
- n_estimators: 100

Setelah diketahui _hyperparameter_ terbaik, maka dapat dilakukan evaluasi terhadap model yang telah diketahui _hyperparameter_ optimal tersebut. Berikut kode yang dapat dijalankan.

```
forest_reg_ht = RandomForestRegressor(max_features=6, n_estimators=100)

train_rmse, test_rmse = evaluation(forest_reg_ht, X_train_prepared, y_train, X_test_prepared, y_test)
```

Meskipun telah dilaksanakan fine-tuning, model Random Forest Regressor secara _default_ memiliki unjuk kerja yang lebih baik.

## Evaluation
Pada bagian ini akan membahas metrik yang digunakan dan hasil _training_ serta evaluasi dari setiap model.

### Formula RMSE

Pada proyek ini metrik evaluasi yang digunakan adalah _Root Mean Squared Error_. Alasan menggunakan RMSE adalah karena metrik ini memiliki presisi yang cukup tinggi dan dataset yang digunakan tidak memiliki _outlier_ sehingga dapat menggunakan RMSE sebagai metrik evaluasi. Rumus RMSE dapat diekspresikan sebagai berikut.

![Formula RMSE](https://miro.medium.com/max/966/1*lqDsPkfXPGen32Uem1PTNg.png)

Keterangan:

- n     : jumlah banyak data
- y_hat : merupakan prediksi model
- y     : merupakan nilai target

### Evaluasi Model

Berikut adalah tabel akhir hasil dari _training_ dan evaluasi dari setiap model.

| Models           | Train_rmse    | Test_rmse   |
| -------------    |:-------------:| -----------:|
| LinearRegression | 1839.376566	 | 1831.351756 |
| DTRegressor      | 1239.391596   | 1256.800549 |
| RFRegressor      | 1013.40798    | 1026.672234 |
| RFRegressorFT    | 1087.930119   | 1050.877457 |

Setelah _training_ dan evaluasi selesai, maka hasil dari setiap model divisualisasikan sebagai berikut.

![bar_chart_modelling](https://github.com/ilhamMalik51/DicodingAppliedML/blob/1abdea6b0d07668e604904332699cc80e989ac7e/Proyek-Pertama/assets/bar_chart_modelling.png)

Berdasarkan hasil di atas dapat diambil kesimpulan bahwa model yang terbaik untuk menjadi solusi adalah **Random Forest Regressor.** Setelah dilakukan berbagai eksperimen, terlihat bahwa untuk memperbaiki nilai metrik tersebut adalah dengan mencoba mengambil data kembali dengan atribut lain. Hal ini dikarenakan dengan atribut yang ada, atribut-atribut tersebut tidak memiliki korelasi linear terhadap nilai target, sehingga model Machine Learning akan kesulitan untuk mencari pola dari atribut-atribut tersebut.

Selain itu, pendekatan lain pun dapat dicoba, seperti menggunakan teknik pendekatan permasalahan _time-series_ mungkin akan menghasilkan yang berbeda bahkan lebih baik.

## Kesimpulan
Untuk menjawab permasalahan pada bagian Problem Statement, dapat diurutkan sebagai berikut:
1. Penyebab Tingginya Traffic Volume dipengaruhi oleh **pukul waktu** dan **temperatur** saat itu.
2. Jika seseorang mengetahui keadaan dan temperatur saat itu, dengan memanfaatkan model Machine Learning, orang tersebut dapat memprediksikan traffic volume pada saat itu juga.
3. Jika ingin meningkatkan unjuk kerja dari model Machine Learning maka perlu adanya perubahan atribut dataset atau pengambilan ulang dataset.
4. Selain itu, terdapat pendekatan lain yang dapat digunakan untuk memprediksikan **traffic_volume** yaitu pendekatan _time-series_.

## Referensi
[1].Bharadwaj, Shashank, Sudheer Ballare, and Munish K. Chandel. "Impact of congestion on greenhouse gas emissions for road transport in Mumbai metropolitan region." Transportation Research Procedia 25 (2017): 3538-3551.

[2].Hopkins, John L., and Judith McKay. "Investigating ‘anywhere working’as a mechanism for alleviating traffic congestion in smart cities." Technological Forecasting and Social Change 142 (2019): 258-272.

[3].Kesuma, P. A., M. A. Rohman, and C. A. Prastyanto. "Risk analysis of traffic congestion due to problem in heavy vehicles: a concept." IOP Conference Series: Materials Science and Engineering. Vol. 650. No. 1. IOP Publishing, 2019.


**---Ini adalah bagian akhir laporan---**


