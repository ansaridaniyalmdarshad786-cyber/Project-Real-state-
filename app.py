import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Real Estate Buyer Intelligence", page_icon="🏠", layout="wide")

@st.cache_data
def load_data():
    df=pd.read_csv("buyer_segmentation_results.csv")
    tx=pd.read_csv("transactions_cleaned.csv")
    return df,tx

df,tx=load_data()

st.title("🏠 Real Estate Buyer Intelligence Dashboard")
st.caption("Machine-learning based buyer segmentation and investment profiling | Unified Mentor project")
st.info("Segments are descriptive, data-driven clusters from the supplied dataset. They are not predictive scores and should be interpreted with the project limitations in mind.")

with st.sidebar:
    st.header("Filters")
    countries=st.multiselect("Country",sorted(df.country.unique()),default=sorted(df.country.unique()))
    regions=st.multiselect("Region",sorted(df.region.unique()),default=sorted(df.region.unique()))
    purposes=st.multiselect("Acquisition purpose",sorted(df.acquisition_purpose.unique()),default=sorted(df.acquisition_purpose.unique()))
    types=st.multiselect("Client type",sorted(df.client_type.unique()),default=sorted(df.client_type.unique()))
    segments=st.multiselect("Buyer segment",sorted(df.segment.unique()),default=sorted(df.segment.unique()))

    f=df[df.country.isin(countries)&df.region.isin(regions)&df.acquisition_purpose.isin(purposes)&df.client_type.isin(types)&df.segment.isin(segments)].copy()

c1,c2,c3,c4=st.columns(4)
c1.metric("Buyers",f.client_id.nunique())
c2.metric("Investment-purpose buyers",int((f.acquisition_purpose=="Investment").sum()))
c3.metric("Loan applicants",int((f.loan_applied=="Yes").sum()))
c4.metric("Avg satisfaction",f.satisfaction_score.mean() if len(f) else 0)

st.subheader("Buyer Segmentation Overview")
seg=f.groupby("segment").agg(Buyers=("client_id","count"),Avg_Age=("age","mean"),Avg_Transactions=("transactions","mean"),Avg_Sale_Price=("avg_sale_price","mean"),Avg_Total_Sales=("total_sales","mean")).reset_index()
col1,col2=st.columns(2)
with col1:
    st.plotly_chart(px.bar(seg,x="segment",y="Buyers",title="Segment Distribution"),use_container_width=True)
with col2:
    st.plotly_chart(px.bar(seg,x="segment",y="Avg_Transactions",title="Average Sold Transactions per Buyer"),use_container_width=True)

st.subheader("Investor Behavior Dashboard")
p=pd.crosstab(f.segment,f.acquisition_purpose,normalize="index").reset_index()
p_m=p.melt(id_vars="segment",var_name="Purpose",value_name="Share")
l=pd.crosstab(f.segment,f.loan_applied,normalize="index").reset_index()
l_m=l.melt(id_vars="segment",var_name="Loan",value_name="Share")
col1,col2=st.columns(2)
with col1:
    st.plotly_chart(px.bar(p_m,x="segment",y="Share",color="Purpose",title="Purpose Mix",labels={"Share":"Share"}),use_container_width=True)
with col2:
    st.plotly_chart(px.bar(l_m,x="segment",y="Share",color="Loan",title="Financing Mix",labels={"Share":"Share"}),use_container_width=True)

st.subheader("Geographic Buyer Analysis")
geo=f.groupby("country").size().reset_index(name="Buyers")
col1,col2=st.columns(2)
with col1:
    st.plotly_chart(px.choropleth(geo,locations="country",locationmode="country names",color="Buyers",title="Buyer Distribution by Country"),use_container_width=True)
with col2:
    reg=f.groupby("region").size().reset_index(name="Buyers").sort_values("Buyers",ascending=False).head(15)
    st.plotly_chart(px.bar(reg,y="region",x="Buyers",orientation="h",title="Top Buyer Regions"),use_container_width=True)

st.subheader("Segment Insights")
for seg_name in sorted(f.segment.unique()):
    d=f[f.segment==seg_name]
    if len(d)==0: continue
    with st.expander(f"{seg_name} — {len(d):,} buyers"):
        a,b,c,dcol=st.columns(4)
        a.metric("Avg age",f"{d.age.mean():.1f}")
        b.metric("Avg transactions",f"{d.transactions.mean():.2f}")
        c.metric("Avg sale price",f"${d.avg_sale_price.mean():,.0f}")
        dcol.metric("Investment purpose",f"{(d.acquisition_purpose.eq('Investment').mean()*100):.1f}%")
        st.write("Top countries:", ", ".join(d.country.value_counts().head(5).index.tolist()))
        st.write("Referral mix:", d.referral_channel.value_counts(normalize=True).mul(100).round(1).astype(str).add("%").to_dict())

st.subheader("Data Preview")
st.dataframe(f[["client_id","client_type","country","region","acquisition_purpose","loan_applied","referral_channel","satisfaction_score","transactions","avg_sale_price","segment"]].head(100),use_container_width=True)
st.download_button("Download filtered buyer data",f.to_csv(index=False).encode("utf-8"),"filtered_buyer_segments.csv","text/csv")
