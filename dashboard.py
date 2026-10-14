from pathlib import Path
from collections import Counter
import json
import pandas as pd
import altair as alt
import streamlit as st
from monitoring import (ROOT,SOURCE,read_csv,read_json,snapshot,compare_results,issue_counts,
                        safe_image,ATTEMPT_COLUMNS,verified_predictions)

local=read_json(ROOT/'local_images.json') or {}
default_data=Path(local.get('data_dir',str(SOURCE/'data/current')))
default_run=Path(local.get('run_dir',str(SOURCE/'runs/current')))

st.set_page_config(page_title='이미지 품질 Harness',page_icon='◈',layout='wide')
st.markdown('''<style>
.block-container {max-width:1420px;padding-top:2rem;}
[data-testid="stMetric"] {background:#f4f7fa;border:1px solid #dde5ed;border-radius:12px;padding:16px;}
[data-testid="stMetricLabel"] {color:#46576c;}
[data-testid="stMetricValue"], [data-testid="stMetricValue"] * {color:#18243a !important;}
[data-testid="stMetricLabel"] * {color:#46576c !important;}
h1,h2,h3 {letter-spacing:-.03em;}
[data-testid="stCaptionContainer"] {font-size:.95rem;line-height:1.65;}
[data-testid="stMetricValue"] {font-size:clamp(1.5rem,3vw,2.3rem);}
[data-baseweb="tab-list"] {gap:1rem;flex-wrap:wrap;}
[data-baseweb="tab"] {height:auto;min-height:48px;white-space:normal;}
</style>''',unsafe_allow_html=True)
st.caption('TEAM PROJECT  /  IMAGE QUALITY HARNESS')
st.title('이미지 품질 Harness')
st.write('검사에서 재시도와 격리까지, 이미지 처리 과정과 CNN 학습 결과를 함께 확인합니다.')

with st.sidebar:
    st.header('프로젝트 결과')
    st.write('이미지 정제 파이프라인과 CNN 성능 비교')
    st.caption('전체 실험 로그·예측 결과를 제공하며, 이미지는 분기와 클래스별 대표 사례만 포함합니다.')
    st.caption('각 그룹에서 파일명 순서로 2장씩 선정했습니다.')
    st.caption('검사·재시도·격리를 제어하는 자체 Python Harness입니다.')
run=SOURCE/'runs/current'
data=SOURCE/'data/current'
results=SOURCE/'results/current'
raw_images=data/'raw/train'
run_images=run
test_images=data/'evaluation/test'
prediction_file=ROOT/'part4_results/test_predictions.csv'
samples=read_json(ROOT/'samples.json')
auto=False
ISSUES={'blur':'흐릿함','dark':'밝기 부족','low_resolution':'낮은 해상도','none':'정상'}
STATUS={'SAVED':'통과·저장','QUARANTINED':'기준 미달·격리'}

def bars(records, category, value, color='#369b95', percent=False):
    frame=pd.DataFrame(records)
    order=frame[category].tolist()
    base=alt.Chart(frame).encode(
        y=alt.Y(f'{category}:N',sort=order,title=None,axis=alt.Axis(labelAngle=0,labelLimit=180)),
        x=alt.X(f'{value}:Q',title=value,scale=alt.Scale(domain=[0,100]) if percent else alt.Scale(zero=True),
                  axis=alt.Axis(format='.0f' if percent else ',d',tickCount=4)),
        tooltip=[alt.Tooltip(f'{category}:N'),alt.Tooltip(f'{value}:Q',format='.2f' if percent else ',d')])
    chart=(base.mark_bar(color=color,size=26)+base.mark_text(align='left',dx=6,fontSize=14).encode(
        text=alt.Text(f'{value}:Q',format='.2f' if percent else ',d'))).properties(height=max(150,len(records)*54))
    st.altair_chart(chart.configure_axis(labelFontSize=13,titleFontSize=13,labelAngle=0).configure_view(strokeWidth=0),width='stretch')

def csv_download(label,rows,filename):
    st.download_button(label,pd.DataFrame(rows).to_csv(index=False).encode('utf-8-sig'),filename,'text/csv')

def image_panel(root,relative,label):
    try:
        path=safe_image(root,relative)
        if path.is_file():st.image(str(path),caption=label,width='stretch')
        else:st.info(f'{label}: 이미지 미포함 또는 미연결')
    except (ValueError,OSError) as exc:st.warning(str(exc))

@st.fragment(run_every=5 if auto else None)
def content():
    try:
        snap=snapshot(run,data)
        comparison=compare_results(results,snap,data)
        attempts,pending=read_csv(run/'validation_attempts.csv',ATTEMPT_COLUMNS) if (run/'validation_attempts.csv').exists() else ([],False)
    except (OSError,ValueError,KeyError,TypeError) as exc:
        st.error(f'데이터를 읽을 수 없습니다: {exc}')
        st.info('폴더 경로와 CSV·JSON 형식을 확인하세요. 쓰기 중이었다면 잠시 후 새로고침하세요.')
        return
    totals=snap['totals']; rows=snap['rows']
    split=Counter(r['split'] for r in snap['manifest'])
    st.subheader('전체 데이터와 실험 구성')
    cols=st.columns(4)
    for col,label,count in zip(cols,['전체 실험 데이터','학습용 · 정제 대상','검증용 · 학습 중 확인','테스트용 · 최종 평가'],
                               [len(snap['manifest']),split['train'],split['val'],split['test']]):
        col.metric(label,f'{count:,}장')
    st.info('약 2만 5천 장 규모의 데이터를 학습·검증·테스트로 나누었습니다. Harness 정제는 학습용 17,451장에 적용했고, 두 CNN의 최종 정확도는 공통 테스트 3,742장으로 비교했습니다.')
    if snap['complete']:st.success('완료 로그 확인 · CSV와 summary 집계 일치')
    elif snap['errors']:st.error('집계 불일치 · 결과를 확정하지 않습니다.')
    else:st.info(snap['state']+' · 로그만으로 프로세스가 현재 동작하는지는 알 수 없습니다.')
    for error in snap['errors']:st.warning(error)
    if snap['pending'] or pending:st.caption('작성 중인 마지막 CSV 행은 다음 새로고침에서 반영됩니다.')
    if snap['progress'] is not None:
        st.progress(min(1.,snap['progress']),text=f"학습용 이미지 정제 완료 {totals['total_images']:,} / {snap['expected']:,}장")
    else:st.caption('Train manifest가 없어 전체 진행률을 계산하지 않습니다.')
    overview,trace,cnn,evidence=st.tabs(['처리 현황','이미지별 제어 흐름','CNN 결과 비교','검증과 해석'])
    with overview:
        cols=st.columns(4)
        st.subheader('학습용 이미지 정제 결과')
        cols[0].metric('정제 대상 · 학습용',f"{totals['total_images']:,}장")
        cols[1].metric('문제 탐지',f"{totals['detected_images']:,}장")
        cols[2].metric('품질 통과 · 저장',f"{totals['saved']:,}장")
        cols[3].metric('기준 미달 · 격리',f"{totals['quarantined']:,}장")
        cols=st.columns(3)
        cols[0].metric('품질 통과율',f"{totals['success_rate']*100:.2f}%" if totals['success_rate'] is not None else '—')
        cols[1].metric('재시도 이미지',f"{totals['retry_images']:,}장")
        cols[2].metric('재시도 후 통과',f"{totals['retry_pass']:,}장")
        st.caption('품질 통과율 = Clean / 처리 기록. CNN 분류 정확도와 다른 지표입니다. 실행 중에는 현재까지 처리된 이미지 기준입니다.')
        st.subheader('검증 결과에 따른 분기')
        flow=[{'경로':'정상 원본 저장','이미지 수':totals['normal_saved']},
              {'경로':'최초 변환 후 통과','이미지 수':totals['first_pass']-totals['normal_saved']},
              {'경로':'재시도 후 통과','이미지 수':totals['retry_pass']},
              {'경로':'최종 격리','이미지 수':totals['quarantined']}]
        bars(flow,'경로','이미지 수')
        left,right=st.columns(2)
        with left:
            st.subheader('문제 유형')
            counts=issue_counts(rows)
            if counts:bars([{'문제':ISSUES.get(k,k),'이미지 수':v} for k,v in counts.items()],'문제','이미지 수',color='#7199d0')
            else:st.info('문제 탐지 기록이 없습니다.')
            st.caption('한 이미지에 여러 문제가 있으면 유형별로 중복 집계됩니다.')
        with right:
            st.subheader('클래스별 최종 처리')
            if rows:
                frame=pd.DataFrame([{'동물':{'cat':'고양이','dog':'개'}.get(r['filename'].split('/')[0]),'상태':STATUS[r['final_status']]} for r in rows])
                counts=frame.groupby(['동물','상태']).size().reset_index(name='이미지 수')
                chart=alt.Chart(counts).mark_bar(size=36).encode(
                    y=alt.Y('동물:N',title=None,axis=alt.Axis(labelAngle=0)),
                    x=alt.X('이미지 수:Q',title='이미지 수',axis=alt.Axis(format=',d',tickCount=4)),
                    color=alt.Color('상태:N',scale=alt.Scale(domain=['통과·저장','기준 미달·격리'],range=['#369b95','#d4924d']),legend=alt.Legend(orient='bottom',title=None,labelFontSize=13)),
                    tooltip=['동물','상태',alt.Tooltip('이미지 수:Q',format=',d')]).properties(height=165)
                st.altair_chart(chart.configure_axis(labelFontSize=13,titleFontSize=13,labelAngle=0).configure_view(strokeWidth=0),width='stretch')
                st.dataframe(counts.pivot(index='동물',columns='상태',values='이미지 수').reindex(columns=['통과·저장','기준 미달·격리']),width='stretch')
        audit,_=read_csv(data/'source_audit.csv')
        if audit:st.caption(f'분할 전 제외 {len(audit):,}개는 위 Harness 격리 수와 별도입니다.')
        csv_download('최종 처리 로그 다운로드',rows,'verification_log.csv')
        st.download_button('현재 집계 다운로드',json.dumps(totals,ensure_ascii=False,indent=2),'monitoring_summary.json','application/json')
    with trace:
        st.subheader('이미지 한 장의 처리 과정을 추적합니다')
        st.write('Extractor → Transformer → Validator → 남은 문제로 Retry → Clean 또는 Quarantine')
        a,b,c=st.columns([2,1,1])
        query=a.text_input('파일명 검색',placeholder='예: cat_10009')
        status=b.selectbox('최종 상태',['전체','SAVED','QUARANTINED'],format_func=lambda x:STATUS.get(x,x))
        issue=c.selectbox('최초 문제',['전체',*sorted(issue_counts(rows))],format_func=lambda x:ISSUES.get(x,x))
        filtered=[r for r in rows if query.lower() in r['filename'].lower() and
                  (status=='전체' or r['final_status']==status) and (issue=='전체' or issue in r['issue'].split('|'))]
        st.caption(f'검색 결과 {len(filtered):,}장 · 아래 표는 최대 200행, 다운로드는 전체 검색 결과입니다.')
        display=pd.DataFrame(filtered[:200]).rename(columns={'filename':'파일명','issue':'최초 문제','validation':'검증','retry_count':'재시도 횟수','final_status':'최종 상태','reason':'판정 사유','output_path':'저장 위치'})
        if not display.empty:
            display['최종 상태']=display['최종 상태'].map(STATUS)
            display['최초 문제']=display['최초 문제'].apply(lambda x:' · '.join(ISSUES.get(v,v) for v in x.split('|')))
        st.dataframe(display,hide_index=True,width='stretch')
        csv_download('검색 결과 다운로드',filtered,'filtered_images.csv')
        filtered=[r for r in filtered if r['filename'] in samples['train']]
        st.caption('상세 이미지 선택에는 웹에 포함된 대표 사례만 표시됩니다. 위 표와 집계는 전체 기록입니다.')
        if filtered:
            options=[r['filename'] for r in filtered]
            default=options.index('cat/cat_10009.png') if 'cat/cat_10009.png' in options else 0
            selected=st.selectbox('상세 조회 이미지',options,index=default)
            row=next(r for r in filtered if r['filename']==selected)
            st.write(f"**{selected}** · {STATUS[row['final_status']]} · 재시도 {row['retry_count']}회")
            st.caption(row['reason'])
            records=[]
            for at in attempts:
                if at['filename']!=selected:continue
                try:measure=json.loads(at['metrics'])
                except (ValueError,TypeError):measure={}
                records.append({'단계':'최초 탐지' if at['attempt']=='-1' else ('최초 검증' if at['attempt']=='0' else f"Retry {at['attempt']}"),
                    '역할':at['agent'],'실행 도구':at['action'],'판정':at['status'],'남은 문제':at['issues'],
                    '밝기':measure.get('brightness'),'선명도 지표':measure.get('blur'),
                    '너비':measure.get('width'),'높이':measure.get('height'),'사유':at['reason']})
            st.dataframe(pd.DataFrame(records),hide_index=True,width='stretch')
            if snap['config']:
                q=snap['config']
                st.caption(f"기준: 밝기 ≥ {q['brightness_threshold']}, 선명도 지표 ≥ {q['blur_threshold']}, 크기 ≥ {q['minimum_width']}×{q['minimum_height']}, 최대 Retry {q['max_retries']}회")
            first,last=st.columns(2)
            with first:image_panel(raw_images,selected,'학습 입력 Raw')
            with last:image_panel(run_images,row['output_path'],'Clean 처리본' if row['final_status']=='SAVED' else '격리 원본 · 처리본이 아닙니다')
            st.caption('실험에 사용한 대표 이미지 원본과 실제 처리 결과입니다.')
    with cnn:
        st.subheader('공통 Test에 대한 실험 결과')
        st.info('Raw 학습 입력에도 RGB 변환과 분할 전 중복 제외가 적용됐으며, Train 일부에는 인위적 품질 저하가 들어 있습니다.')
        if comparison['available']:
            a,b,c=st.columns(3)
            a.metric('Raw CNN Test 정확도',f"{comparison['raw']['test_accuracy']*100:.2f}%")
            b.metric('Harness CNN Test 정확도',f"{comparison['harness']['test_accuracy']*100:.2f}%")
            c.metric('정확도 차이',f"{comparison['difference_pp']:+.2f}%p")
            bars([{'모델':'정제 전 CNN','정확도 (%)':comparison['raw']['test_accuracy']*100},
                  {'모델':'정제 후 CNN','정확도 (%)':comparison['harness']['test_accuracy']*100}], '모델','정확도 (%)',percent=True)
            histories={k:pd.DataFrame(read_csv(results/f'{k}_history.csv')[0]) for k in ['raw','harness']}
            for metric,label in [('accuracy','정확도'),('loss','손실')]:
                series={}
                for k,df in histories.items():
                    if not df.empty and {metric,'val_'+metric,'epoch'}.issubset(df.columns):
                        for col,suffix in [(metric,'학습'),('val_'+metric,'검증')]:
                            series[{'raw':'정제 전','harness':'정제 후'}[k]+' · '+suffix]=pd.Series(pd.to_numeric(df[col]).values,index=pd.to_numeric(df['epoch']).values)
                if series:
                    st.write(f'**에포크별 {label}**')
                    chart=pd.DataFrame(series);chart.index.name='에포크'
                    st.line_chart(chart)
            st.caption('최종 5번째 에포크 모델의 단일 seed 결과입니다. 데이터 수와 업데이트 수가 달라 정제만의 인과 효과나 일반적 향상을 확정할 수 없습니다. ')
        else:
            st.warning(comparison.get('reason','결과 비교 차단'))
            for error in comparison['errors']:st.error(error)
        try:
            predictions,prediction_error=verified_predictions(prediction_file,results,snap['manifest'])
        except (ValueError,OSError,KeyError) as exc:
            predictions,prediction_error=[],str(exc)
        if predictions and comparison['available']:
            st.success(f'Test {len(predictions):,}장 사전 재추론 기록 검증 · 기존 정확도 재현')
            st.subheader('같은 이미지에서 달라진 예측')
            outcomes=Counter(('둘 다 정답' if r['raw_prediction']==r['true_label'] and r['harness_prediction']==r['true_label'] else
                'Raw만 정답' if r['raw_prediction']==r['true_label'] else 'Harness만 정답' if r['harness_prediction']==r['true_label'] else '둘 다 오답') for r in predictions)
            st.dataframe(pd.DataFrame([{'구분':k,'이미지 수':v} for k,v in outcomes.items()]),hide_index=True,width='stretch')
            cols=st.columns(2)
            for k,col in zip(['raw','harness'],cols):
                with col:
                    st.write(f'**{k.title()} 혼동행렬**')
                    table=pd.crosstab(pd.Series([r['true_label'] for r in predictions],name='실제'),pd.Series([r[k+'_prediction'] for r in predictions],name='예측'))
                    st.dataframe(table.reindex(index=['cat','dog'],columns=['cat','dog'],fill_value=0),width='stretch')
            mode=st.selectbox('예측 사례 필터',['전체','Harness만 정답','Raw만 정답','둘 다 오답'])
            matches=[r for r in predictions if mode=='전체' or
                (mode=='Harness만 정답' and r['harness_prediction']==r['true_label'] and r['raw_prediction']!=r['true_label']) or
                (mode=='Raw만 정답' and r['raw_prediction']==r['true_label'] and r['harness_prediction']!=r['true_label']) or
                (mode=='둘 다 오답' and r['raw_prediction']!=r['true_label'] and r['harness_prediction']!=r['true_label'])]
            matches=[r for r in matches if r['filename'] in samples['test']]
            st.caption('이미지 선택은 대표 사례이며, 위 비교와 혼동행렬은 전체 Test 기준입니다.')
            if matches:
                selected_test=st.selectbox('Test 이미지 선택',[r['filename'] for r in matches])
                example=next(r for r in matches if r['filename']==selected_test)
                pic,details=st.columns([1,2])
                with pic:image_panel(test_images,selected_test,'두 모델에 동일하게 입력한 Test 이미지')
                with details:
                    st.write(f"**정답: {example['true_label']}**")
                    for k in ['raw','harness']:
                        st.write(f"{k.title()} 예측: **{example[k+'_prediction']}** · dog 확률 {float(example[k+'_dog_probability'])*100:.2f}%")
                    st.caption('확률은 이 모델의 출력값이며 객관적인 확실성을 보장하지 않습니다.')
            csv_download('Test 예측 전체 다운로드',predictions,'test_predictions.csv')
        elif prediction_error:st.info(f'예측 사례 표시 보류: {prediction_error}')
    with evidence:
        st.subheader('확인한 것과 아직 확인하지 못한 것')
        st.write('**확인**: 최종 처리 로그 집계, Retry 분기, 저장·격리 수, 모델/데이터 manifest 해시, 전달된 A/B 조건의 일치.')
        image_check=read_json(ROOT/'image_validation.json')
        if image_check:
            st.write(f"**배포 전 전체 이미지 검사: {image_check['status']}** · Clean {image_check['clean_revalidated']:,}장 재검증 · 격리 원본 {image_check['quarantine_original_checked']:,}장 대조")
            st.dataframe(pd.DataFrame(image_check['coverage']).T,width='stretch')
            st.caption('전체 원본 데이터에 수행한 검사 기록입니다. 이 웹사이트에는 대표 이미지 48개 파일만 포함되며 학습과 재추론을 새로 실행하지 않습니다.')
            st.caption('공개 manifest는 개인 PC 경로 열을 제거한 사본입니다. 원본 및 공개 사본의 해시를 별도로 기록해 구분합니다.')
        st.write('**해석의 한계**: 저해상도 확대·샤프닝은 잃어버린 정보 복원을 보장하지 않습니다. 수치 기준 통과와 사람이 판단하는 이미지 품질은 다를 수 있습니다.')
        st.write('**Harness 구현**: 역할별 Python 객체가 검사·변환·검증을 수행하고, Orchestrator가 남은 문제에 따라 재시도와 격리를 결정합니다. 단순히 함수 이름에 Agent를 붙인 사실만으로 판단하지 않고 시도 로그에서 동작을 추적합니다.')
        with st.expander('품질 검사 설정'):
            st.json(snap['config'])
        with st.expander('인위적 품질 저하 기록'):
            corruptions=Counter(r['corruption'] for r in snap['manifest'] if r['split']=='train')
            st.dataframe(pd.DataFrame([{'유형':k,'이미지 수':v} for k,v in corruptions.items()]),hide_index=True)
        st.caption('실행 시간·처리 속도는 원본 로그에 타임스탬프가 없어 추정하여 표시하지 않습니다.')

content()
