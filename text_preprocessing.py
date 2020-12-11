import re

#text = re.sub(r'[^\x00-\x7F]+', '', text)

class text_preprocessing:

    def __init__(self, text):
        self.text = text

    def remove_URL(self):
        ## Removes url 
        text = re.sub(r"pic.twitter\S+", "_url_", self.text)
        text = re.sub(r"\S+\.com\S*", "_url_", text)
        text = re.sub(r"\S+\.fr\S*", "_url_", text)
        text = re.sub(r"\S+\.tv\S*", "_url_", text)
        text = re.sub(r"\S+\.be\S*", "_url_", text)
        text = re.sub(r"\S+\.me\S*", "_url_", text)
        text = re.sub(r"\S+\.ly\S*", "_url_", text)
        text = re.sub(r"\S*\d{6,}\S*", "", text)
        text = re.sub(r"http\S+", "_url_", text)
        return text #re.sub(r"([_url_ ]\s*){2,}", "_url_", text)
    
    def remove_useless_spaces(self):
        # removes double or triple space
        text = re.sub(' +', ' ', self.text)
        text.replace(" ,", ",")
        text.replace(" .", ".")

        return text

    def to_lowercase(self):
        text = ""
        for letter in self.text:
            new_letter = letter.lower()
            text = text + new_letter
        return text
    
    def remove_blank(self):
        return re.sub(r"\s", " ", self.text)
    
    def remove_multiple_commas(self):
        text = self.text.replace(",,,", ",")
        text.replace(",,", ",")
        return text

    def remove_emojis(self):
        regrex_pattern = re.compile(pattern = "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
                            "]+")
        return regrex_pattern.sub(r'',self.text)
    
    def remove_hashtags(self):
        return re.sub(r"#\w+", "", self.text)
    
    def remove_mentions(self):
        return re.sub(r"@\w+", "_user_", self.text)

    def clean_mispelling(self):
        abv = {'t': 'tu es', 't\'es' : 'tu es', 'ki' : 'qui', 'fdp' : 'fils de pute', 'pk' : 'pourquoi', 'jvai' : 'je vais', 'jvais' : 'je vais', 'd\'la' : 'de la',
        'sal' : 'sale', 'mm' : 'meme', 'c' : 'c\'est', 'mdr' : 'mort de rire', 'ptdr' : 'mort de rire', 'qq' : 'quelque', 'qqe' : 'quelque', 'koi' : 'quoi',
        'oklm' : 'au calme', 'jsp' : 'je ne sais pas', 'blc' : 'bas les couilles', 'jpp' : 'j\'en peux plus' , 'msk' : 'miskine', 'wsh' : 'wesh' , 'wlh' : 'wallah' ,
        'askip' : 'a ce qu\'il parait', 'bjr' : 'bonjour', 'bcp' : 'beaucoup' , 'ajd' : 'aujourd\'hui' , 'tt' : 'tout', 'cho' : 'chaud', 'pg' : 'pas grave' , 'dac' : 'd\'accord',
        'ct' : 'c\'etait', 'dsl' : 'desole', 'g' : 'j\'ai', 'chui' : 'je suis', 'dab' : 'd\'habitude', 'é' : 'et', 'twa': 'toi', 'jtm' : 'je t\'aime', 'jv' : 'je vais',
        'jve' : 'je veux' , 'jveu' : 'je veux', 'jveux' : 'je veux', 'kdo' : 'cadeau', 'kan' : 'quand' , 'kand' : 'quand', 'ke' : 'que', 'kel': 'quel',
        'kelle': 'quelle', 'kestu' : 'qu\'est ce que tu', 'lakl' : 'laquelle', 'lekl' : 'lequel', 'lu': 'salut', 'alu' : 'salut', 'msg': 'message', 'now' : 'maintenant',
        'mtn' : 'maintenant', 'nn' : 'non', 'nptk' : 'n\'importe quoi', 'nrv' : 'enerve', 'vnr' : 'enerve', 'nsp' : 'ne sais pas', 'o' : 'au', 'osef' : 'on s\'en fou',
        'psq' : 'parce que', 'parske':'parce que', 'pkoi' : 'pourquoi', 'keske' : 'qu\'est ce que', 'queske' : 'qu\'est ce que', 'qqch' : 'quelque chose', 'qch' : 'quelque chose',
        'raf' : 'rien a faire', 'rdv': 'rendez vous', 're' : 'rebonjour', 'ras' : 'rien a signaler', 'ri1' : 'rien', 'r': 'rien', 'sava' : 'ca va', 'sv': 'ca va' , 'slt' : 'salut',
        'stp' : 's\'il te plait', 'svp': 's\'il vous plait', 'tfk' : 'tu fais quoi', 'tg' : 'ta gueule', 'tjr' : 'toujours', 'tjs' : 'toujours', 'tjrs' : 'toujours',
        'tlm' : 'tout le monde', 'tps' : 'temps', 'vzi' : 'vas y', 'vazi' : 'vas y', 'vdm' : 'vie de merde', 'vrm' : 'vraiment', 'vrmt' : 'vraiment'  }

        words = self.text.split()
        reformed = [abv[word] if word in abv else word for word in words]
        return( " ".join(reformed))
    
    def format_text(self):
        self.text = self.remove_blank()
        self.text = self.to_lowercase()
        self.text = self.remove_multiple_commas()
        self.text = self.remove_useless_spaces()
        self.text = self.remove_emojis()
        self.text = self.remove_hashtags()
        self.text = self.remove_mentions()
        self.text = self.remove_URL()
        self.text = self.clean_mispelling()
        return self.text

def split_on_number(text):
    # split our text by number of post (used with reddit and gab posts dataset)
    text = re.sub(r"1. ", "", text)
    split_text = []
    i = 2
    while bool(re.search(str(i)+'. ', text)):
        regex = re.compile((str(i)+". "))
        text_list = re.split(regex, text)
        split_text.append(text_list[0])
        text = text_list[1]
        i +=1
        
    split_text.append(text)
    return split_text


def clean_text(text):
    # Taken from the previous work on MLMA hate speech -> https://github.com/HKUST-KnowComp/MLMA_hate_speech/blob/master/annotated_data_processing.py
    """
        text: a string
        
        return: modified initial string
    """
    replace_by_blank_symbols = re.compile('\u00bb|\u00a0|\u00d7|\u00a3|\u00eb|\u00fb|\u00fb|\u00f4|\u00c7|\u00ab|\u00a0\ude4c|\udf99|\udfc1|\ude1b|\ude22|\u200b|\u2b07|\uddd0|\ude02|\ud83d|\u2026|\u201c|\udfe2|\u2018|\ude2a|\ud83c|\u2018|\u201d|\u201c|\udc69|\udc97|\ud83e|\udd18|\udffb|\ude2d|\udc80|\ud83e|\udd2a|\ud83e|\udd26|\u200d|\u2642|\ufe0f|\u25b7|\u25c1|\ud83e|\udd26|\udffd|\u200d|\u2642|\ufe0f|\udd21|\ude12|\ud83e|\udd14|\ude03|\ude03|\ude03|\ude1c|\udd81|\ude03|\ude10|\u2728|\udf7f|\ude48|\udc4d|\udffb|\udc47|\ude11|\udd26|\udffe|\u200d|\u2642|\ufe0f|\udd37|\ude44|\udffb|\u200d|\u2640|\udd23|\u2764|\ufe0f|\udc93|\udffc|\u2800|\u275b|\u275c|\udd37|\udffd|\u200d|\u2640|\ufe0f|\u2764|\ude48|\u2728|\ude05|\udc40|\udf8a|\u203c|\u266a|\u203c|\u2744|\u2665|\u23f0|\udea2|\u26a1|\u2022|\u25e1|\uff3f|\u2665|\u270b|\u270a|\udca6|\u203c|\u270c|\u270b|\u270a|\ude14|\u263a|\udf08|\u2753|\udd28|\u20ac|\u266b|\ude35|\ude1a|\u2622|\u263a|\ude09|\udd20|\udd15|\ude08|\udd2c|\ude21|\ude2b|\ude18|\udd25|\udc83|\ude24|\udc3e|\udd95|\udc96|\ude0f|\udc46|\udc4a|\udc7b|\udca8|\udec5|\udca8|\udd94|\ude08|\udca3|\ude2b|\ude24|\ude23|\ude16|\udd8d|\ude06|\ude09|\udd2b|\ude00|\udd95|\ude0d|\udc9e|\udca9|\udf33|\udc0b|\ude21|\udde3|\ude37|\udd2c|\ude21|\ude09|\ude39|\ude42|\ude41|\udc96|\udd24|\udf4f|\ude2b|\ude4a|\udf69|\udd2e|\ude09|\ude01|\udcf7|\ude2f|\ude21|\ude28|\ude43|\udc4a|\uddfa|\uddf2|\udc4a|\ude95|\ude0d|\udf39|\udded|\uddf7|\udded|\udd2c|\udd4a|\udc48|\udc42|\udc41|\udc43|\udc4c|\udd11|\ude0f|\ude29|\ude15|\ude18|\ude01|\udd2d|\ude43|\udd1d|\ude2e|\ude29|\ude00|\ude1f|\udd71|\uddf8|\ude20|\udc4a|\udeab|\udd19|\ude29|\udd42|\udc4a|\udc96|\ude08|\ude0d|\udc43|\udff3|\udc13|\ude0f|\udc4f|\udff9|\udd1d|\udc4a|\udc95|\udcaf|\udd12|\udd95|\udd38|\ude01|\ude2c|\udc49|\ude01|\udf89|\udc36|\ude0f|\udfff|\udd29|\udc4f|\ude0a|\ude1e|\udd2d|\uff46|\uff41|\uff54|\uff45|\uffe3|\u300a|\u300b|\u2708|\u2044|\u25d5|\u273f|\udc8b|\udc8d|\udc51|\udd8b|\udd54|\udc81|\udd80|\uded1|\udd27|\udc4b|\udc8b|\udc51|\udd90|\ude0e')
    replace_by_apostrophe_symbol = re.compile('\u2019')
    replace_by_dash_symbol = re.compile('\u2014')
    replace_by_u_symbols = re.compile('\u00fb|\u00f9')
    replace_by_a_symbols = re.compile('\u00e2|\u00e0') 
    replace_by_c_symbols = re.compile('\u00e7') 
    replace_by_i_symbols = re.compile('\u00ee|\u00ef') 
    replace_by_o_symbols = re.compile('\u00f4') 
    replace_by_oe_symbols = re.compile('\u0153')
    replace_by_e_symbols = re.compile('\u00e9|\u00ea|\u0117|\u00e8')
    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|,;]')
    text = replace_by_e_symbols.sub('e', text)
    text = replace_by_a_symbols.sub('a', text)
    text = replace_by_o_symbols.sub('o', text)
    text = replace_by_oe_symbols.sub('oe', text)
    text = replace_by_u_symbols.sub('e', text)
    text = replace_by_i_symbols.sub('e', text)
    text = replace_by_u_symbols.sub('e', text)
    text = replace_by_apostrophe_symbol.sub("'", text)
    text = replace_by_dash_symbol.sub("_", text)
    text = replace_by_blank_symbols.sub('', text)