
#********************** Kullanılan Kütüphaneler **********************# 
import numpy as np
import pandas as pd
import seaborn as sns
import nltk 
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix 
from sklearn import metrics
import string
from nltk.corpus import stopwords
nltk.download('stopwords')
pd.options.mode.chained_assignment = None

#*********************** data *****************************# 
Data = pd.read_csv('complaints.csv',low_memory=False )

################### ön işleme ###############################3
dt = Data[["narrative"]]
dt["narrative"] = dt["narrative"].astype(str)
dt["text_lower"] = dt["narrative"].str.lower()
dt.drop(["text_lower"], axis=1 , inplace=True)

PUNCT_TO_REMOVE = string.punctuation
def noktalamaSil(text):
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))
dt["text_data"] = dt["narrative"].apply(lambda text: noktalamaSil(text))

STOPWORDS = set(stopwords.words('english'))
def stopwordsSil(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])
dt["text_wo_stop"] = dt["text_data"].apply(lambda text: stopwordsSil(text))

#######################################################################
#print(Data)
print(Data.dtypes)
print(pd.notnull(Data['narrative']).value_counts()) ## işe yaramayan kullanılmayacak olan ögeler elenir
col = ['product', 'narrative']
Data = Data[col] 
print(Data.columns) ##data için 
Data.columns = ['product', 'narrative'] 
Data['category_id'] = Data['product'].factorize()[0]
category_id_df = Data[['product', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'product']].values)

print ("------------------------------------ Data Gruplandırması --------------------------------------------------")
print(Data.groupby('product').narrative.count())
print ("------------------------------------ Data  --------------------------------------------------")
print(Data.head(10))

fig = plt.figure(figsize=(8,6))
Data.groupby('product').narrative.count().plot.bar(ylim=0)
plt.show()

###########################  Count vect - Tf-Idf vect ############## 
print ("-------------------------- Count vect - Tf-Idf vect --------------------------------")
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern='\w{1,}', max_features=5000)    
tfidf_vect.fit(Data['narrative'].astype('U').values)
Features = tfidf_vect.transform(Data['narrative'].astype('U').values)
encoder = preprocessing.LabelEncoder()
Labels1 = encoder.fit_transform(Data['product'])
print(Features.shape)
print(Features[0], Labels1)
print("\n")

#################################  filtreleme  ######################################### 
N = 2
for product, category_id in sorted(category_to_id.items()):
  features_chi2 = chi2(Features, Labels1 == category_id) 
  indices = np.argsort(features_chi2[0])
  feature_names = np.array(tfidf_vect.get_feature_names_out())[indices]
  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
  print("# '{}':".format(product))
  print("  . En ilişkili unigramlar:\n       . {}".format('\n       . '.join(unigrams[-N:])))
  print("  . En ilişkili biagramlar:\n       . {}".format('\n       . '.join(bigrams[-N:]))) 


############################## Öğrenme Traning Count vect - Tf-Idf vect - MultinomialNB ################################ 

print ("--------------------------------------------------------------------------------------")
X_train, X_test, y_train, y_test = train_test_split(Data['narrative'], Data['product'], random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform((X_train).astype('U').values)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform((X_train_counts).astype('U'))
clf = MultinomialNB().fit(X_train_tfidf, y_train)

################################################## Test  ############################################
print(clf.predict(count_vect.transform(["itfaiye kurtarıldı deneyimli hasar sigorta şirketi hızlı gönderdi iyi iş muayene penceresi erimiş hasarlı takip ipotek sahibi teftiş takip onaylı muayene tamamlandı baskı işi satıcı tarafından tamamlandı ödenen iş sigorta şirketi muayene gönderdi ferforje çit işi le yıkıcı satıcı her zaman geç muayene penceresini tamamladı doğrulanmış izleme ofis, emanet fonu iç hesabını izlemek istedi, her biri geri ödemeye ihtiyaç duyduğunda işlem yapmak, aileye gerekli yükü, özellikle zaman, ev borcunu koymak, ödeme süresini izlemek, emanet hesabında para saklamak, genel gider, stresli zaman, sebep, Mülk sahipliği yüzdesi göz önüne alındığında, iyi bir ödeme geçmişi."])))
print(clf.predict(count_vect.transform(["Promosyon Tarihi Üç Sayı İncele Comenity Bank Perakende Kartı Dolandırıcılık Comenity Bank Dolandırıcı Banka Kredi Kartı Sağlayıcısı Çocukların Yeri New York Sonsuza Kadar Victoria Secret Ev Kredisi Comenity Bank, Gecikmelerle Alt Limitler Almaya Başlıyor aşırı maliyet ücretleri. ayrıca kart keşif limiti de değişti aşırı ödeme gecikmesi şirket banka kredi limitini değiştireceğini söyledi ilk limite ulaşıldı, en yüksek ödeme hesabının etkilendiğini söyledi istikrarlı kredi kartı kredi notu keskin bir şekilde olumsuz düştü Şu anda borcumu ödüyorum miktar yolsuzluk borcu olumsuz etkiledi tedarik şirketi kredili mevduat hesabı çok geç ücretler ilk kredi limiti çok dalgalansa bile şirket büyük bir komisyon alıyor hesabı düzeltmek istiyorum kızgın avukat sebeplerini söyledi a bana ulaşın ben bir şirket çalışanıyım yardım isim iletişim bi ilginiz için teşekkürler"])))
Data[Data['narrative'] == " itfaiye kurtarıldı deneyimli hasar sigorta şirketi hızlı gönderdi iyi iş muayene penceresi erimiş hasarlı takip ipotek sahibi teftiş takip onaylı muayene tamamlandı baskı işi satıcı tarafından tamamlandı ödenen iş sigorta şirketi muayene gönderdi ferforje çit işi le yıkıcı satıcı her zaman geç muayene penceresini tamamladı doğrulanmış izleme ofis, emanet fonu iç hesabını izlemek istedi, her biri geri ödemeye ihtiyaç duyduğunda işlem yapmak, aileye gerekli yükü, özellikle zaman, ev borcunu koymak, ödeme süresini izlemek, emanet hesabında para saklamak, genel gider, stresli zaman, sebep, Mülk sahipliği yüzdesi göz önüne alındığında, iyi bir ödeme geçmişi."]
Data[Data['narrative'] == "Promosyon Tarihi Üç Sayı İncele Comenity Bank Perakende Kartı Dolandırıcılık Comenity Bank Dolandırıcı Banka Kredi Kartı Sağlayıcısı Çocukların Yeri New York Sonsuza Kadar Victoria Secret Ev Kredisi Comenity Bank, Gecikmelerle Alt Limitler Almaya Başlıyor aşırı maliyet ücretleri. ayrıca kart keşif limiti de değişti aşırı ödeme gecikmesi şirket banka kredi limitini değiştireceğini söyledi ilk limite ulaşıldı, en yüksek ödeme hesabının etkilendiğini söyledi istikrarlı kredi kartı kredi notu keskin bir şekilde olumsuz düştü Şu anda borcumu ödüyorum miktar yolsuzluk borcu olumsuz etkiledi tedarik şirketi kredili mevduat hesabı çok geç ücretler ilk kredi limiti çok dalgalansa bile şirket büyük bir komisyon alıyor hesabı düzeltmek istiyorum kızgın avukat sebeplerini söyledi a bana ulaşın ben bir şirket çalışanıyım yardım isim iletişim bi ilginiz için teşekkürler"]

############################### Model kontrol ve değerlendirme ############################################
print ("######################################## Model #######################################")
models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0),
]

CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
  data_google_model = model.__class__.__name__
  accuracies = cross_val_score(model, Features, Labels1, scoring='accuracy', cv=CV)
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((data_google_model, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['data_google_model', 'fold_idx', 'accuracy'])

############################ tablo ##################### 
sns.boxplot(x='data_google_model', y='accuracy', data=cv_df)
sns.stripplot(x='data_google_model', y='accuracy', data=cv_df, 
              size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.show()
print(cv_df.groupby('data_google_model').accuracy.mean()) 
print ("#######################################################################################")

########################## conf-matris ########################################

model = LinearSVC()

X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(Features, Labels1, Data.index, test_size=0.33, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test) 

conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=Data['product'].unique(),yticklabels=Data['product'].unique())
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show() 

print(model.fit(Features, Labels1)) 
print(metrics.classification_report(y_test, y_pred, target_names=Data['product'].unique()))

