import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import re


def get_mysql_engine():
    """Database bağlantısı için fonksiyon"""
    # Database bağlantısı için .env dosyasını yükle
    load_dotenv()

    # .env dosyasındaki DATABASE_URL'yi kullan
    direct_url = os.getenv("DATABASE_URL")
    direct_url = "mysql+pymysql://root:root@localhost:3306/ULAKBIM-CABIM-UBYT-bs"

    if not direct_url:
        raise RuntimeError("DATABASE_URL bulunamadı. Lütfen .env veya sistem ortamında ayarlayın.")
    database_url = direct_url

    engine = create_engine(
        database_url,
        pool_pre_ping=True,
        pool_recycle=3600,
        pool_size=5,
        max_overflow=10,
        future=True,
    )
    return engine


def test_connection_and_counts():
    """Bağlantıyı test et, tabloları listele ve tablo satır sayılarını yazdır"""
    engine = get_mysql_engine()
    with engine.connect() as conn:
        # Tabloları listele
        result = conn.execute(text("SHOW TABLES"))
        tables = [row[0] for row in result]
        print("Tables in bs_cec:", tables)

        # Tabloların satır sayılarını yazdır
        for table in [
            "cec_user",
            "cec_article_raw",
            "cec_citation",
            "cec_citation_intent",
        ]:
            if table in tables:
                count = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                print(f"{table}: {count.scalar()} rows")


def apply_square_bracket_rules(s: str) -> str:
    # 1) Köşeli atıflar → <CITE>  Örn: "[12]", "[3-5]", "[8, 9]"
    s = re.sub(r"\[(?=[^\]]*\d)[^\]]*\]", "<CITE>", s)
    # 2) Satır içinde açılıp devam eden köşeli → <CITE>  Örn: "[8, "
    s = re.sub(r"\[\s*\d+\s*,\s*", "<CITE>", s)
    # 3) Satır içinde kapanan köşeli → <CITE>  Örn: "15]"
    s = re.sub(r"\s*\d+\s*\]", "<CITE>", s)
    # 3.1) Bozuk zincir: sayı + virgül + nokta → <CITE>.  Örn: "5, ." → "<CITE>."
    s = re.sub(r"\b\d+\s*,\s*\.", "<CITE>.", s)
    # 3.2) Parantezsiz "vd." + <CITE> → <CITE>  Örn: "Nahzat vd. [31]" → "<CITE>"
    s = re.sub(r"\b[A-Za-zÇĞİÖŞÜçğıöşü][\w .’'\\-]{0,60}\s+vd\.?\s*<CITE>", "<CITE>", s)
    # 3.3) Köşeli içinde bozuk "Soyad, , YIL;" zinciri (bir veya daha fazla) → <CITE>
    #      Örn: "[Aksoy, , 1990; Hepçilingirler, , 1999;" → "<CITE>"
    s = re.sub(r"\[\s*(?:[^\],;]{1,60}\s*,\s*,\s*[12]\d{3}[a-z]?\s*;\s*)+\]?", "<CITE>", s)
    return s


def apply_parenthetical_rules(s: str) -> str:
    # 4) Parantezli et al./vd., yıl; (noktalı virgül zinciri) → <CITE>  Örn: "(Scherer vd. 2015;"
    s = re.sub(r"\([^)]*?\b(?:vd\.|et al\.)\s*,?\s*[12]\d{3}\s*;\s*\)?", "<CITE>", s)
    # 5) Parantezli et al./vd., yıl harf listesi → <CITE>  Örn: "(Augot et al., 2017a,b)"
    s = re.sub(r"\([^)]*?\b(?:et al\.|vd\.)\s*,\s*[12]\d{3}[a-z](?:\s*,\s*[a-z])+\s*\)", "<CITE>", s)
    # 6) Parantezli et al./vd., yıl harf eki → <CITE>  Örn: "(Augot et al., 2017a)"
    s = re.sub(r"\([^)]*?\b(?:et al\.|vd\.)\s*,\s*[12]\d{3}[a-z]\s*\)", "<CITE>", s)
    # 6.a) Parantezli çoklu yazarlar (virgüller) + 've' + yıl → <CITE>
    s = re.sub(
        r"\(\s*[A-Za-zÇĞİÖŞÜçğıöşü][\w .’'\\-]{0,60}(?:\s*,\s*[A-Za-zÇĞİÖŞÜçğıöşü][\w .’'\\-]{0,60})*\s*,\s*ve\s+[A-Za-zÇĞİÖŞÜçğıöşü][\w .’'\\-]{0,60}\s+[12]\d{3}\s*\)",
        "<CITE>", s)
    # 6.b) Aynı yapı ama yıldan önce virgül var → <CITE>
    s = re.sub(
        r"\(\s*[A-Za-zÇĞİÖŞÜçğıöşü][\w .’'\\-]{0,60}\s+ve\s+[A-Za-zÇĞİÖŞÜçğıöşü][\w .’'\\-]{0,60}\s*,\s*[12]\d{3}\s*\)?",
        "<CITE>", s)
    # 6.c) Çoklu yazarlar + 've' + yıl + noktalı virgül → <CITE>
    s = re.sub(
        r"\(\s*[A-Za-zÇĞİÖŞÜçğıöşü][\w .’'\\-]{0,60}(?:\s*,\s*[A-Za-zÇĞİÖŞÜçğıöşü][\w .’'\\-]{0,60})*\s*,\s*ve\s+[A-Za-zÇĞİÖŞÜçğıöşü][\w .’'\\-]{0,60}\s+[12]\d{3}\s*;\s*\)?",
        "<CITE>", s)
    # 6.e) Parantezli iki yazar ("ve") + yıl (virgülsüz) → <CITE>
    s = re.sub(r"\(\s*[^\s,()]{1,40}\s+ve\s+[^\s,()]{1,40}\s+[12]\d{3}\s*\)", "<CITE>", s)
    # 6.f) Parantezli iki yazar ("ve") + yıl + ; (kapanış opsiyonel) → <CITE>  Örn: "(Bäuml ve Stiefelhagen 2011;)"
    s = re.sub(r"\(\s*[^\s,()]{1,40}\s+ve\s+[^\s,()]{1,40}\s+[12]\d{3}\s*;\s*\)?", "<CITE>", s)
    # 6.h) Parantezli "v. diğ.," + yıl + ); → <CITE>  Örn: "(Glowinski v. diğ., 1995)"
    s = re.sub(r"\(\s*[A-ZÇĞİÖŞÜ][A-Za-zÇĞİÖŞÜçğıöşü'’\.-]{1,30}\s+v\.\s+diğ\.,\s*[12]\d{3}\s*[;)]\s*\)?", "<CITE>", s)
    # 6.d) Parantezli iki yazar ("ve") + noktalı virgül (yıl yok) → <CITE>
    s = re.sub(
        r"\(\s*[A-ZÇĞİÖŞÜ][A-Za-zÇĞİÖŞÜçğıöşü'’\-\.]{1,30}\s+ve\s+[A-ZÇĞİÖŞÜ][A-Za-zÇĞİÖŞÜçğıöşü'’\-\.]{1,30}\s*;\s*\)?",
        "<CITE>", s)
    # 6.d) Parantezli ", vd.," + yıl → <CITE>
    s = re.sub(r"\(\s*[A-Za-zÇĞİÖŞÜçğıöşü][A-Za-zÇĞİÖŞÜçğıöşü .'\-]*,\s*vd\.?\s*,\s*[12]\d{3}\s*\)", "<CITE>", s)
    # 6.d.1) Parantezli ", vd." + yıl (vd. sonrası virgül yok) → <CITE>  Örn: "(Sgrò, vd. 2010)"
    s = re.sub(r"\(\s*[^,()]{1,60},\s*vd\.?\s*[12]\d{3}\s*\)", "<CITE>", s)
    # 6.i) Parantezli tek yazar + "ve Ark.," + yıl) → <CITE>  Örn: "(Saad ve Ark., 2013)"
    s = re.sub(r"\(\s*[A-ZÇĞİÖŞÜ][A-Za-zÇĞİÖŞÜçğıöşü'’\-]{1,30}\s+ve\s+[Aa]rk\.,\s*[12]\d{3}\s*\)", "<CITE>", s)
    # 7) Parantezli vd. virgüllü yıl → <CITE>
    s = re.sub(r"\(\s*[A-Za-zÇĞİÖŞÜçğıöşü][A-Za-zÇĞİÖŞÜçğıöşü .'\-]*\s+vd\.?\s*,\s*[12]\d{3,4}[a-z]?\s*\)", "<CITE>", s)
    # 7.b) Parantezli vd. boşluklu yıl (virgülsüz) → <CITE>
    s = re.sub(r"\(\s*[A-Za-zÇĞİÖŞÜçğıöşü][A-Za-zÇĞİÖŞÜçğıöşü .'\-]{0,60}\s+vd\.?\s*[12]\d{3}(?:\s*[;,)])?", "<CITE>",
               s)
    # 7.a) Parantezli vd. virgüllü yıl + ; → <CITE>
    s = re.sub(r"\(\s*[A-Za-zÇĞİÖŞÜçğıöşü][A-Za-zÇĞİÖŞÜçğıöşü .'\-]*\s+vd\.?\s*,\s*[12]\d{3,4}[a-z]?\s*;\s*\)?",
               "<CITE>", s)
    # 8) Parantezli et al. virgüllü yıl → <CITE>
    s = re.sub(r"\(\s*[A-Za-zÇĞİÖŞÜçğıöşü][A-Za-zÇĞİÖŞÜçğıöşü .'\-]*\s+et al\.?\s*,\s*\d{4}\s*\)", "<CITE>", s)
    # 9) Parantezli sayfalı yazar-yıl → <CITE>
    s = re.sub(r"\(\s*[A-Za-zÇĞİÖŞÜçğıöşü][A-Za-zÇĞİÖŞÜçğıöşü .'\-]*,\s*[12]\d{3}\s*:\s*\d+(?:\s*[-–]\s*\d+)?\s*\)",
               "<CITE>", s)
    # 9.c) Parantezli yazar, yıl, s. sayfa → <CITE>  Örn: "(Lavigna, 2002, s. 369)"
    s = re.sub(
        r"\(\s*[A-Za-zÇĞİÖŞÜçğıöşü][A-Za-zÇĞİÖŞÜçğıöşü .'\-]*,\s*[12]\d{3}\s*,\s*s\.?\s*\d+(?:\s*[-–]\s*\d+)?\s*\)",
        "<CITE>", s)
    # 9.a) Parantezli 'vd./v.d., :sayfa' + ; veya ) → <CITE>  Örn: "(VanDyne vd., :1367;" veya ")"
    s = re.sub(r"\(\s*[^()]{0,80}\b(?:vd|v\.d)\.?\s*,?\s*:\s*\d+(?:\s*[-–]\s*\d+)?\s*(?:;|\))\s*\)?", "<CITE>", s)
    # 9.b) Parantezli üç isim (ilkinden sonra virgül) + '&' + yıl (bozuk boşluklar dahil) → <CITE>
    #     Örn: "(Zhang, Luo& Zhang2015)"
    s = re.sub(r"\(\s*[^\s,()]{1,40}\s*,\s*[^\s,()]{1,40}\s*&\s*[^\s,()]{1,40}\s*[12]\d{3}\s*\)", "<CITE>", s)
    # 10) Parantezli yazar-yıl (virgüllü) → <CITE>
    s = re.sub(r"\(\s*[A-Za-zÇĞİÖŞÜçğıöşü][A-Za-zÇĞİÖŞÜçğıöşü .'\-]*?,\s*[12]\d{3}\s*\)", "<CITE>", s)
    # 10.c) Parantezli yazar, yıl+harf → <CITE>
    s = re.sub(r"\(\s*[A-Za-zÇĞİÖŞÜçğıöşü][A-Za-zÇĞİÖŞÜçğıöşü .'\-]*?,\s*[12]\d{3}[a-z]\s*\)", "<CITE>", s)
    # 10.d) Parantezli yazar, yıl(+harf) + ; → <CITE>
    s = re.sub(r"\(\s*[A-Za-zÇĞİÖŞÜçğıöşü][A-Za-zÇĞİÖŞÜçğıöşü .'\-]*?,\s*[12]\d{3}[a-z]?\s*;\s*\)?", "<CITE>", s)
    # 10.a) Parantezli yazar-yıl (çift virgül, fazla kapanış) → <CITE>
    s = re.sub(r"\(\s*[A-Za-zÇĞİÖŞÜçğıöşü][A-Za-zÇĞİÖŞÜçğıöşü .'\-]*?,\s*,\s*[12]\d{3}\s*\)+", "<CITE>", s)
    # 11) Parantezli et al./vd. boşluklu yıl/genel → <CITE>
    s = re.sub(r"\(\s*[A-Za-zÇĞİÖŞÜçğıöşü][A-Za-zÇĞİÖŞÜçğıöşü .,’'\-,]{0,60}(?:\s+et al\.|\s+vd\.)?\s+[12]\d{3}\s*\)",
               "<CITE>", s)
    # 11.a) Parantezli çoklu yazar (virgüller) + yıl + noktalı virgül → <CITE>
    s = re.sub(
        r"\(\s*[A-Za-zÇĞİÖŞÜçğıöşü][A-Za-zÇĞİÖŞÜçğıöşü .,’'\-]{0,40}(?:\s*,\s*[A-Za-zÇĞİÖŞÜçğıöşü][A-Za-zÇĞİÖŞÜçğıöşü .,’'\-]{0,40})*\s+[12]\d{3}\s*;\s*\)?",
        "<CITE>", s)
    # 11.f) Parantezli et al./vd., n.d. → <CITE>
    s = re.sub(r"\(\s*[A-Za-zÇĞİÖŞÜçğıöşü][^)]*?\b(?:et al\.|vd\.)\s*,\s*n\.d\.\s*\)", "<CITE>", s)
    # 11.g) Parantezli iki yazar & n.d. → <CITE>
    s = re.sub(r"\(\s*[^\s,&;()]{1,40}\s&\s[^\s,&;()]{1,40}\s*,\s*n\.d\.\s*\)", "<CITE>", s)
    # 11.b) Parantezli çoklu yazar(virgüller) + & + n.d. + ; → <CITE>
    s = re.sub(r"\(\s*[^,;()]{1,60}(?:\s*,\s*[^,;()]{1,60})*\s*,?\s*&\s*[^,;()]{1,60}\s*,\s*n\.d\.\s*;\s*\)?", "<CITE>",
               s)
    # 12) Parantezli zincir bozuk kapanış → <CITE>
    s = re.sub(r"\(\s*[A-Za-zÇĞİÖŞÜçğıöşü][A-Za-zÇĞİÖŞÜçğıöşü .'\-]*,\s*\d{4}\s*;\s*\)?", "<CITE>", s)
    # 13) Parantezli et al./vd. + yıl + sonda virgül → <CITE>
    s = re.sub(r"\(\s*[A-Za-zÇĞİÖŞÜçğıöşü][A-Za-zÇĞİÖŞÜçğıöşü .'\-]*\s+(?:et al\.|vd\.)\s+[12]\d{3}\s*,\s*\)", "<CITE>",
               s)
    # 14) Parantezli iki yazar & yıl (virgüllü) → <CITE>  Örn: "(J. Zhang & Sanderson, 2009)"
    s = re.sub(r"\(\s*[^\s,&;()]{1,40}\s&\s[^\s,&;()]{1,40},\s*[12]\d{3}\s*\)", "<CITE>", s)
    # 14.b) Parantezli iki yazar & yıl (virgülsüz) → <CITE>  Örn: "(Gamba & Houshmand 2000)"
    s = re.sub(r"\(\s*[^\s,&;()]{1,40}\s&\s[^\s,&;()]{1,40}\s+[12]\d{3}\s*\)", "<CITE>", s)
    # 14.a) Parantezli iki yazar & yıl + ; (kapanış opsiyonel) → <CITE>  Örn: "(Fernandez-Llatas & García-Gómez, 2014;"
    s = re.sub(r"\(\s*[^\s,&;()]{1,40}\s&\s[^\s,&;()]{1,40},\s*[12]\d{3}\s*;\s*\)?", "<CITE>", s)
    # 14.c) Parantezli iki yazar & yıl + ; (virgülsüz) → <CITE>  Örn: "(Maas & Vosselman 1999; )"
    s = re.sub(r"\(\s*[^\s,&;()]{1,40}\s&\s[^\s,&;()]{1,40}\s+[12]\d{3}\s*;\s*\)?", "<CITE>", s)
    # 14.d) Parantezli çoklu yazarlar + & + yıl + ; opsiyonel → <CITE>  Örn: "(Çelık, Yıldız, & Karadenız, 2019;"
    s = re.sub(r"\(\s*[^,;()]{1,60}(?:\s*,\s*[^,;()]{1,60})*\s*,?\s*&\s*[^,;()]{1,60}\s*,\s*[12]\d{3}\s*;?\s*\)?",
               "<CITE>", s)
    # 15) Parantezli yazar başharfli biçim → <CITE>  Örn: "Soyad, N.P., (1990)"
    s = re.sub(r"\b[A-Za-zÇĞİÖŞÜçğıöşü][A-Za-zÇĞİÖŞÜçğıöşü\-'\.]{1,30},\s*(?:[A-Z]\.){1,3},\s*\([12]\d{3}\)\s*\.?",
               "<CITE>", s)
    # 16) Parantezli soyad (yıl) (riskli) — DISABLED (çok geniş, özel kurallarla çakışıyor)
    # 16a) Tek yazar (yıl) — DISABLED (genel kurallarla çakışıyor)
    # 16b) "ve" ile iki yazar (yıl) — DISABLED (özel kurallar mevcut)
    # 16.t) İki tek kelime yazar '&' ile + (yıl[harf]?) → <CITE>  Örn: "Zhang & Zhang (2006)"
    s = re.sub(
        r"\b[A-ZÇĞİÖŞÜ][A-Za-zÇĞİÖŞÜçğıöşü'’\-\.]{1,30}\s*&\s*[A-ZÇĞİÖŞÜ][A-Za-zÇĞİÖŞÜçğıöşü'’\-\.]{1,30}\s*\([12]\d{3}[a-z]?\)",
        "<CITE>", s)
    # 16.s) Tek kelime Soyad + (yıl[harf]?) → <CITE>  Örn: "Lam (2007)", "Kayran (1996a)"
    s = re.sub(r"\b[A-ZÇĞİÖŞÜ][A-Za-zÇĞİÖŞÜçğıöşü'’\-\.]{1,30}\s*\([12]\d{3}[a-z]?\)", "<CITE>", s)
    # 17) Parantezli et al./vd. + yıl (kapanış eksik, satır sonu) → <CITE>  Örn: "(Luo et al. 2018"
    s = re.sub(
        r"\(\s*[A-Za-zÇĞİÖŞÜçğıöşü][A-Za-zÇĞİÖŞÜçğıöşü .’'\\-]{0,60}(?:\s+et al\.|\s+vd\.)\s+[12]\d{3}\s*(?:[;,.])?\s*$",
        "<CITE>", s)
    return s


def apply_non_parenthetical_rules(s: str) -> str:
    # 18) Parantezsiz iki yazar & yıl) → <CITE>  Örn: "Holat & Kulaç, 2014)"
    s = re.sub(
        r"\b[A-Za-zÇĞİÖŞÜçğıöşü][\w .’'\\-]{0,60}\s&\s[A-Za-zÇĞİÖŞÜçğıöşü][\w .’'\\-]{0,60}\s*,\s*[12]\d{3}\s*\)",
        "<CITE>", s)
    # 19) Parantezsiz iki yazar ("ve") + yıl) → <CITE>  Örn: "Irani ve Love, 2002)"
    s = re.sub(
        r"\b[A-ZÇĞİÖŞÜ][A-Za-zÇĞİÖŞÜçğıöşü'’-]{1,30}\s+ve\s+[A-ZÇĞİÖŞÜ][A-Za-zÇĞİÖŞÜçğıöşü'’-]{1,30}\s*,\s*[12]\d{3}\s*\)",
        "<CITE>", s)
    # 19.a) Parantezsiz iki yazar ("ve") + yıl) (virgülsüz) → <CITE>  Örn: "Ntalampiras ve Fakotakis 2012)"
    s = re.sub(
        r"\b[A-ZÇĞİÖŞÜ][A-Za-zÇĞİÖŞÜçğıöşü'’-]{1,30}\s+ve\s+[A-ZÇĞİÖŞÜ][A-Za-zÇĞİÖŞÜçğıöşü'’-]{1,30}\s+[12]\d{3}\s*\)",
        "<CITE>", s)
    # 20) Parantezsiz yazar, yıl(+harf)) → <CITE>  Örn: "Bacon, 1991)", "Hekimoglu, 2019a)"
    s = re.sub(r"\b[A-ZÇĞİÖŞÜ][A-Za-zÇĞİÖŞÜçğıöşü'’-]{1,30}\s*,\s*[12]\d{3}[a-z]?\s*\)", "<CITE>", s)
    # 21) Parantezsiz yazar, yıl; → <CITE>  Örn: "Kabasakal, 2018;"
    s = re.sub(
        r"\b[A-ZÇĞİÖŞÜ][A-Za-zÇĞİÖŞÜçğıöşü'’-]{1,30}(?:\s+(?:ve|&)\s+[A-ZÇĞİÖŞÜ][A-Za-zÇĞİÖŞÜçğıöşü'’-]{1,30})?\s*,\s*[12]\d{3}\s*;",
        "<CITE>", s)
    # 22) Parantezsiz iki yazar ("ve") + yıl + ";" → <CITE>  Örn: "Goudbeek ve Scherer 2010;"
    s = re.sub(
        r"\b[A-Za-zÇĞİÖŞÜçğıöşü][\w .’'\\-]{0,60}\s+ve\s+[A-Za-zÇĞİÖŞÜçğıöşü][\w .’'\\-]{0,60}\s*,?\s*[12]\d{3}\s*;",
        "<CITE>", s)
    # 22.b) Parantezsiz iki yazar ("and") + yıl(+harf) + ";" veya ")" → <CITE>  Örn: "Sarode and Mandaogade 2014b;"
    s = re.sub(
        r"\b[A-ZÇĞİÖŞÜ][A-Za-zÇĞİÖŞÜçğıöşü'’\.\-]{1,60}\s+and\s+[A-ZÇĞİÖŞÜ][A-Za-zÇĞİÖŞÜçğıöşü'’\.\-]{1,60}\s*,?\s*[12]\d{3}[a-z]?\s*[;)]",
        "<CITE>", s)
    # 22.a) Parantezsiz tek yazar + "et al." + yıl + ";" → <CITE>  Örn: "Liu et al. 2018;"
    s = re.sub(r"\b[A-ZÇĞİÖŞÜ][A-Za-zÇĞİÖŞÜçğıöşü'’\.\-]{1,30}\s+et al\.\s*,?\s*[12]\d{3}[a-z]?\s*;", "<CITE>", s)
    # 22.b) Parantezsiz tek yazar + "et al." + yıl → <CITE>  Örn: "Roy et al. 2019"
    s = re.sub(r"\b[A-ZÇĞİÖŞÜ][A-Za-zÇĞİÖŞÜçğıöşü'’\.\-]{1,30}\s+et al\.\s*,?\s*[12]\d{3}[a-z]?\b", "<CITE>", s)
    # 22.d) Parantezsiz tek yazar + yıl(+harf) + ;/ ) → <CITE>  Örn: "Kak 2019;)"
    s = re.sub(r"\b[A-ZÇĞİÖŞÜ][A-Za-zÇĞİÖŞÜçğıöşü'’\.\-]{1,60}\s+[12]\d{3}[a-z]?\s*[;)]", "<CITE>", s)
    # 23) Parantezsiz et al./vd. + yıl(+harf) + ";" → <CITE>  Örn: "Eryiğit et al., 2006b;"
    s = re.sub(r"\b[A-Za-zÇĞİÖŞÜçğıöşü][\w .’'\\-]{0,60}\s+(?:et al\.|vd\.)\s*,\s*[12]\d{3}[a-z]?\s*;", "<CITE>", s)
    # 23.e) Parantezsiz 'Soyad, vd., YIL;' → <CITE>  Örn: "Dhillon, vd., 2017;" (v.d. varyantını da destekler)
    s = re.sub(r"\b[A-ZÇĞİÖŞÜ][A-Za-zÇĞİÖŞÜçğıöşü'’\.\-]{1,60}\s*,\s*(?:vd|v\.d)\.?\s*,\s*[12]\d{3}[a-z]?\s*;",
               "<CITE>", s)
    # 23.a) Parantezsiz vd., (YIL) → <CITE>  Örn: "Grimm vd., (2007)"
    s = re.sub(r"\b[A-Za-zÇĞİÖŞÜçğıöşü][\w\-'\. ]{1,50}\s+vd\.?\s*,\s*\([12]\d{3}\)", "<CITE>", s)
    # 23.b) Parantezsiz vd. (YIL) → <CITE>  Örn: "Chan vd. (2004)"
    s = re.sub(r"\b[A-Za-zÇĞİÖŞÜçğıöşü][A-Za-zÇĞİÖŞÜçğıöşü\-'\.]{0,50}\s+vd\.?\s*\([12]\d{3}\)", "<CITE>", s)
    # 23.b.1) Parantezsiz et al. (YIL+harf) → <CITE>  Örn: "Tasgetiren et al. (2013a)"
    s = re.sub(r"\b[^\W\d_][\w'’\.\-]{0,50}\s+et al\s*\.?\s*\([12]\d{3}[a-z]?\)", "<CITE>", s)
    # 23.c) Parantezsiz vd. (YIL) + space → <CITE>  Örn: "Aydoğdu vd. (2010) ve"
    s = re.sub(r"\b[A-Za-zÇĞİÖŞÜçğıöşü][A-Za-zÇĞİÖŞÜçğıöşü\-'\.]{0,50}\s+vd\.?\s*\([12]\d{3}\)\s+", "<CITE> ", s)
    # 23.d) Parantezsiz iki yazar ("ve") + yıl + space → <CITE>  Örn: "Lam ve Chan (1998) çalışmaları"
    s = re.sub(
        r"\b[A-Za-zÇĞİÖŞÜçğıöşü][A-Za-zÇĞİÖŞÜçğıöşü\-'\.]{0,50}\s+ve\s+[A-Za-zÇĞİÖŞÜçğıöşü][A-Za-zÇĞİÖŞÜçğıöşü\-'\.]{0,50}\s+\([12]\d{3}\)\s+",
        "<CITE> ", s)
    # 24) Parantezsiz et al./vd. + yıl) → <CITE>  Örn: "Chen et al. 2018)"
    s = re.sub(r"\b[A-Za-zÇĞİÖŞÜçğıöşü][\w .’'\\-]{0,60}\s+(?:et al\.|vd\.)\s*,?\s*[12]\d{3}\s*\)", "<CITE>", s)
    # 24.c) Parantezsiz tek yazar + "et al." + yıl + ")" → <CITE>  Örn: "Liu et al. 2018b)"
    s = re.sub(r"\b[A-ZÇĞİÖŞÜ][A-Za-zÇĞİÖŞÜçğıöşü'’\.\-]{1,30}\s+et al\.\s*,?\s*[12]\d{3}[a-z]?\s*\)", "<CITE>", s)
    # 24.d) Parantezsiz "Soyad v. diğ., YIL" (+ opsiyonel )/;) → <CITE>  Örn: "Glowinski v. diğ., 1995)"
    s = re.sub(r"\b[A-ZÇĞİÖŞÜ][A-Za-zÇĞİÖŞÜçğıöşü'’\.\-]{1,30}\s+v\.\s+diğ\.,\s*[12]\d{3}\s*[;)]?", "<CITE>", s)
    # 24.e) Parantezsiz "Soyad vd. YIL" + ; veya ) → <CITE>  Örn: "Vu vd. 2009; )"
    s = re.sub(r"\b[A-ZÇĞİÖŞÜ][A-Za-zÇĞİÖŞÜçğıöşü'’\.\-]{1,60}\s+vd\.?\s*,?\s*[12]\d{3}[a-z]?\s*(?:[;)]\s*)+", "<CITE>",
               s)
    # 24.f) Parantezsiz "Soyad, vd., YIL)" → <CITE>  Örn: "Nadernejad, vd., 2008)"
    s = re.sub(r"\b[A-ZÇĞİÖŞÜ][A-Za-zÇĞİÖŞÜçğıöşü'’\.\-]{1,60}\s*,\s*vd\.?\s*,\s*[12]\d{3}[a-z]?\s*\)", "<CITE>", s)
    # 24.b) Parantezsiz vd., yıl(+harf) → <CITE>  Örn: "Kong vd., 2016b"
    s = re.sub(r"\b[A-ZÇĞİÖŞÜ][A-Za-zÇĞİÖŞÜçğıöşü'’\-\.]{0,50}\s+vd\.?\s*,\s*[12]\d{3}[a-z]?\b", "<CITE>", s)
    # 24.a) Parantezsiz "Soyad, et al., YIL)" → <CITE>  Örn: "Chang, et al., 2017)"
    s = re.sub(r"\b[A-ZÇĞİÖŞÜ][A-Za-zÇĞİÖŞÜçğıöşü'’\-\.]{1,30}\s*,\s*et al\.,\s*[12]\d{3}\s*\)", "<CITE>", s)
    # 25) Parantezsiz vd. + yıl: sayfalı ve ) → <CITE>  Örn: "Güyer vd., 2009: 758)"
    s = re.sub(r"\b[A-Za-zÇĞİÖŞÜçğıöşü][\w .'’\-]{0,60}(?:\s+vd\.)?\s*,\s*[12]\d{3}\s*:\s*\d+(?:\s*[-–]\s*\d+)?\s*\)",
               "<CITE>", s)
    # 26) Parantezsiz vd. + yıl: sayfalı (vd. zorunlu) → <CITE>  Örn: "Fisher vd., 2016:168)"
    s = re.sub(r"\b[A-Za-zÇĞİÖŞÜçğıöşü][\w .'’\-]{0,60}\s+vd\.\s*,\s*[12]\d{3}\s*:\s*\d+(?:\s*[-–]\s*\d+)?\s*\)",
               "<CITE>", s)
    # 26.a) Parantezsiz tek yazar + "ve ark./Ark.," + yıl) → <CITE>  Örn: "Kee ve Ark., 2017)"
    s = re.sub(r"\b[A-ZÇĞİÖŞÜ][A-Za-zÇĞİÖŞÜçğıöşü'’\-]{1,30}\s+ve\s+[Aa]rk\.?,\s*[12]\d{3}\s*\)", "<CITE>", s)
    # 26.b) Parantezsiz tek yazar + "ve ark./Ark." + yıl) (virgülsüz) → <CITE>  Örn: "Kimmel ve Ark. 1997)"
    s = re.sub(r"\b[A-ZÇĞİÖŞÜ][A-Za-zÇĞİÖŞÜçğıöşü'’\-]{1,30}\s+ve\s+[Aa]rk\.?\,?\s*[12]\d{3}\s*\)", "<CITE>", s)
    # 26.c) Parantezsiz tek yazar + "ve ark./Ark.," + yıl + . → <CITE>  Örn: "Can ve Ark., 2012."
    s = re.sub(r"\b[A-ZÇĞİÖŞÜ][A-Za-zÇĞİÖŞÜçğıöşü'’\-]{1,30}\s+ve\s+[Aa]rk\.?,\s*[12]\d{3}\s*\.(?=\s|$)", "<CITE>", s)
    # 26.d) Parantezsiz tek yazar + "ve ark./Ark.," + yıl (sonda noktalama yok) → <CITE>
    #       Örn: "Carrillo ve ark., 2015"
    s = re.sub(r"\b[A-ZÇĞİÖŞÜ][A-Za-zÇĞİÖŞÜçğıöşü'’\-]{1,30}\s+ve\s+[Aa]rk\.?,\s*[12]\d{3}\b", "<CITE>", s)
    # 26.e) Parantezsiz tek yazar + "ve ark./Ark.," + (YIL) → <CITE>
    #       Örn: "Nalçakan ve ark., (2015)", "Ramadan ve ark., (2012)"
    s = re.sub(r"\b[A-ZÇĞİÖŞÜ][A-Za-zÇĞİÖŞÜçğıöşü'’\-]{1,30}\s+ve\s+[Aa]rk\.?,\s*\([12]\d{3}\)\s*", "<CITE>", s)
    # 26.f) Parantezsiz bozuk 'Soyad, , YIL;' → <CITE>  Örn: "Hepçilingirler, , 1999;"
    s = re.sub(r"\b[^,;()]{1,60}\s*,\s*,\s*[12]\d{3}[a-z]?\s*;", "<CITE>", s)
    return s


def apply_broken_parenthesis_rules(s: str) -> str:
    # 27) Parantez açılmış ama kapanmamış: "(Clemons, 1991" → <CITE>
    s = re.sub(r"\(\s*[A-Za-zÇĞİÖŞÜçğıöşü][A-Za-zÇĞİÖŞÜçğıöşü .'\-]*\s*,\s*[12]\d{3}\s*$", "<CITE>", s)
    return s


def replace_citation(batch_size: int = 1000):
    """Databasedeki atıfları <CITE> ile değiştirme"""
    engine = get_mysql_engine()
    pat_digit_brackets = re.compile(r"\[(?=[^\]]*\d)[^\]]*\]")
    last_id = 0
    total_updated = 0

    while True:
        with engine.connect() as conn:
            rows = conn.execute(
                text("""
                     SELECT id, citation_context
                     FROM cec_citation
                     WHERE id > :last_id
                       AND state IS NULL
                       AND (section <> 'Kaynaklar' OR section IS NULL)
                     ORDER BY id LIMIT :lim
                     """),
                {"last_id": last_id, "lim": batch_size},
            ).mappings().all()

        if not rows:
            break

        with engine.begin() as conn:
            for r in rows:
                ctx = r["citation_context"] or ""

                # Normalize NBSP-like spaces BEFORE other rules
                s = re.sub(r'[\u00A0\u2007\u202F]', ' ', ctx)
                # Kurallar yalnızca state IS NULL ve section ≠ 'Kaynaklar/Kaynakça' için uygulanır.

                # --- KÖŞELİ ATIFLAR ---
                s = apply_square_bracket_rules(s)

                # --- PARANTEZ İÇİ (spesifikten genele) ---
                s = apply_parenthetical_rules(s)

                # --- PARANTEZ DIŞI (yazar-yıl çeşitleri) ---
                s = apply_non_parenthetical_rules(s)

                # --- BOZUK PARANTEZ KAPANIŞLARI ---
                s = apply_broken_parenthesis_rules(s)


                if s != ctx:
                    conn.execute(
                        text(
                            "UPDATE cec_citation SET citation_context_clean = :val WHERE id = :id"),
                        {"val": s, "id": r["id"]},
                    )
                    total_updated += 1
                last_id = r["id"]

        print(f"Processed up to id={last_id}, updated={total_updated}")

    print(f"Done. Total updated: {total_updated}")


if __name__ == "__main__":
    replace_citation()
