#####################################################
# PROJE:Kural Tabanlı Sınıflandırma ile Potansiyel Müşteri Getirisi Hesaplama
#####################################################

#####################################################
# İş Problemi : Bir oyun şirketi müşterilerinin bazı özelliklerini kullanarak seviye
# tabanlı (level based) yeni müşteri tanımları (persona) oluşturmak ve bu yeni müşteri
# tanımlarına göre segmentler oluşturup bu segmentlere göre yeni gelebilecek müşterilerin
# şirkete ortalama ne kadar kazandırabileceğini tahmin etmek istemektedir.
#####################################################

#####################################################
# Veri Seti Hikayesi: Persona.csv veri seti uluslararası bir oyun şirketinin sattığı ürünlerin
# fiyatlarını ve bu ürünleri satın alan kullanıcıların bazı demografik bilgilerini barındırmaktadır. Veri
# seti her satış işleminde oluşan kayıtlardan meydana gelmektedir. Bunun anlamı
# tablo tekilleştirilmemiştir. Diğer bir ifade ile belirli demografik özelliklere sahip bir
# kullanıcı birden fazla alışveriş yapmış olabilir.
#####################################################

#####################################################
# Degiskenler:
# Price: Müşterinin harcama tutarı
# Source: Müşterinin bağlandığı cihaz türü
# Sex: Müşterinin cinsiyeti
# Country: Müşterinin ülkesi
# Age: Müşterinin yaşı
#####################################################

#GOREV 1: Aşağıdaki Soruları Yanıtlayınız.

# Soru 1: persona.csv dosyasını okutunuz ve veri seti ile ilgili genel bilgileri gösteriniz

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = pd.read_csv("pythonProgramlama/python_for_data_science/data_analysis_with_python/datasets/persona.csv")
df.head()
df.info()# Hiç boş değer olmadığınız gözlemliyoruz.
df.shape
df.columns
df.index
df.describe().T
# Soru 2: Kaç unique SOURCE vardır? Frekansları nedir?

df["SOURCE"].nunique()
df["SOURCE"].value_counts()

# Soru 3: Kaç unique PRICE vardır?

df["PRICE"].nunique()

# Soru 4: Hangi PRICE'dan kaçar tane satış gerçekleşmiş?

df["PRICE"].value_counts()

# Soru 5: Hangi ülkeden kaçar tane satış olmuş?

df["COUNTRY"].value_counts()

# Soru 6: Ülkelere göre satışlardan toplam ne kadar kazanılmış?

df.groupby("COUNTRY").agg({"PRICE": "sum"})

# Soru 7: SOURCE türlerine göre satış sayıları nedir?

df["SOURCE"].value_counts()

# Soru 8: Ülkelere göre PRICE ortalamaları nedir?

df.groupby(["COUNTRY"]).agg({"PRICE": "mean"})

# Soru 9: SOURCE'lara göre PRICE ortalamaları nedir?
df.groupby(["SOURCE"]).agg({"PRICE": "mean"})

# Soru 10: COUNTRY-SOURCE kırılımında PRICE ortalamaları nedir?
df.groupby(["COUNTRY", "SOURCE"]).agg({"PRICE": "mean"})

# GOREV 2: COUNTRY, SOURCE, SEX, AGE kırılımında ortalama kazançlar nedir?

df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE": "mean"})

# GOREV 3: Çıktıyı PRICE’a göre sıralayınız.

agg_df = df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE": "mean"}).sort_values("PRICE", ascending=False)


#Görev 4: Indekste yer alan isimleri değişken ismine çeviriniz.

agg_df.head()
agg_df = agg_df.reset_index()
agg_df


# Görev 5: Age değişkenini kategorik değişkene çeviriniz ve agg_df’e ekleyiniz.

df["AGE"].nunique()
agg_df["AGE"].max()
intervals = [0,18,25,40,67]
new_labels = ["0_18", "19_25","26_40","41_67"]
agg_df["AGE_CAT"] = pd.cut(agg_df["AGE"], intervals, labels=new_labels)
agg_df.head()

# Görev 6: Yeni seviye tabanlı müşterileri (persona) tanımlayınız.

agg_df["customers_level_based"] = agg_df[["COUNTRY", "SOURCE", "SEX", "AGE_CAT"]].agg(lambda x: "_".join(x).upper(), axis=1)
agg_df = agg_df[["customers_level_based", "PRICE"]]
agg_df.head()

agg_df["customers_level_based"].value_counts() #Bir veriden birden çok olduğunu görebiliriz.

agg_df = agg_df.groupby("customers_level_based").agg({"PRICE": "mean"}) #segmentleri tekillestirmek için groupby islemi yapılıyor.

agg_df = agg_df.reset_index() #Yukarıdaki islemin ardından indexe döndüğü için yeniden degisken haline getiriyoruz.
agg_df.head()

# Kontrol
agg_df["customers_level_based"].value_counts()
agg_df.head()

# Görev 7: Yeni müşterileri (personaları) segmentlere ayırınız.

agg_df["SEGMENT"] = pd.qcut(agg_df["PRICE"], 4, labels=["D", "C", "B", "A"])
agg_df.head()

agg_df.groupby("SEGMENT").agg({"PRICE": ["mean", "max", "sum"]})

# Görev 8: Yeni gelen müşterileri sınıflandırıp, ne kadar gelir getirebileceklerini tahmin ediniz.

# 33 yaşında ANDROID kullanan bir Türk kadını hangi segmente aittir ve ortalama ne kadar gelir kazandırması beklenir?

new_user = "TUR_ANDROID_FEMALE_26_40"
agg_df[agg_df["customers_level_based"] == new_user]

# 35 yaşında IOS kullanan bir Fransız kadını hangi segmente aittir ve ortalama ne kadar gelir kazandırması beklenir?

new_user_2 = "FRA_IOS_FEMALE_26_40"
agg_df[agg_df["customers_level_based"] == new_user_2]



