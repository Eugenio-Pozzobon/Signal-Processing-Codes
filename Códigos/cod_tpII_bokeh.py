import numpy as np
from scipy.signal import square, sawtooth
from scipy.io import wavfile as wav
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
import sounddevice as sd

## -- Bokeh Engine
from bokeh.layouts import row, column, layout
from bokeh.plotting import figure
from bokeh.server.server import Server
from bokeh.models.widgets import Button
from bokeh.models import Slider, Dropdown
from bokeh.layouts import gridplot

# Downsample the signal, selecting every nth element.
def downsample(input, factor):
    output = []
    c=(factor-1)
    for i in input:
        c += 1
        if(c> (factor-1)):
            c=0
            output.append(i)
    return output

# Upsample the signal, inserting n-1 zeros between every element.
def upsample(input, factor):
    output = np.zeros(len(input)*factor)
    c=0
    for i in input:
        output[c*factor] = i
        c+=1

    return output

def amostragem(sig_in,td,ts):
    global seconds
    factor = int(ts/td)
    s_out=downsample(sig_in,factor)
    s_out=upsample(s_out,factor)

    return s_out

#--------
def plot_audio():
    #função para plotagem específica dos gráficos de áudio
    seconds = 1
    fsrec = 44100  # Sample rate
    samplerate = fsrec
    td = 1/samplerate
    t = np.linspace(0, seconds, seconds * samplerate)
    myrecording = sd.rec(int(seconds * fsrec), samplerate=fsrec, channels=2)
    sd.wait()  # Wait until recording is finished
    write('record.wav', fsrec, myrecording)  # Save as WAV file
    rtrec, dtrec = wav.read('record.wav')
    xsig = dtrec[:, 1]
    Lsig=len(xsig)

    ts=1/fsrec # taxa de amostragem 50 Hz
    fator=ts/td
    # envia o sinal a um amostrador
    s_out = xsig #aamostragem(xsig,td,ts)

    # calcula a transformada de Fourier
    Lfft = int(pow(2, np.ceil(np.log2(Lsig)+1)))
    Fmax = int(1 / (2 * td))
    Faxis = np.linspace(-Fmax, Fmax, int(Lfft))
    Xsig = np.fft.fftshift(np.fft.fft(xsig,Lfft))
    S_out = np.fft.fftshift(np.fft.fft(s_out,Lfft))

    # calcula o sinal reconstruído a partir de amostragem ideal e LPF (filtro passa-baixas) ideal
    BW=5000#floor((Lfft/Nfactor)/2)
    H_lpf=np.zeros(Lfft)

    H_lpf[int(Lfft/2 - BW):int(Lfft/2+BW-1)]=1 # LPF ideal
    S_recv=fator*S_out*H_lpf # filtragem ideal
    s_recv=np.real(np.fft.ifft(np.fft.fftshift(S_recv))) # domínio da freq. reconstruído
    s_recv=s_recv[0:Lsig] # domínio do tempo reconstruçdo

    #matplotlib engine
    # traça gráfico do sinal original e do sinal amostrado nos domínios do tempo e da frequencia
    plt.figure(figsize=(10, 10))
    plt.plot(t,xsig[0:Lsig],'k')
    plt.vlines(t,0,s_out[0:Lsig], colors='green')
    plt.xlabel('tempo,segundos')
    plt.title('sinal g(t) e suas amostras uniformes')
    plt.show()

    plt.plot(Faxis,abs(Xsig))
    plt.xlabel('frequência (Hz)')
    plt.title('Espectro de g(t)')
    plt.show()

    plt.plot(Faxis,abs(S_out))
    plt.xlabel('frequência (Hz)')
    plt.title('Espectro de g_T(t)')
    plt.show()

    # traça gráfico do sinal reconstruído idealmente nos domínios do tempo e da frequencia
    plt.plot(Faxis,abs(S_recv))
    plt.xlabel('frequência (Hz)')
    plt.title('Espectro de filtragem ideal (reconstrucao)')
    plt.show()

    write('record2.wav', fsrec, s_recv)

    plt.plot(t,xsig,'k-.')
    plt.plot(t,s_recv,'b')
    plt.xlabel('tempo,segundos')
    plt.title('Sinal original versus Sinal reconstruido idealmente')
    plt.show()

def tpiishow(doc):
    # função que roda o servidor bokeh para plotagem
    global td, seconds, samplerate, t
    global Lfft, Fmax, Faxis, Xsig, S_out
    global signal_freq, bw_freq_slider, xsig, Lsig
    global samplerate2, ts, fator, s_out
    global s_recv, S_recv
    global wavet
    global bw_auto

    #define os valores iniciais
    bw_auto = False

    wavet = "START"

    fator = 0.
    Lsig = 0
    t = []
    xsig = []
    s_out = []
    s_recv = []

    samplerate2 = 50  # intervalo entre os pontos do sinal "analógico"
    signal_freq = 1
    bw_freq_slider = 3

    def calc():
        #realiza o calculo das funções para criar os gráficos. Atualiza as variáveis globais.
        global wavet
        global td, seconds, samplerate, t
        global Lfft, Fmax, Faxis, Xsig, S_out
        global signal_freq, bw_freq_slider, xsig, Lsig
        global samplerate2, ts, fator, s_out
        global s_recv, S_recv, BW

        samplerate = 500  # intervalo entre os pontos do sinal "analógico"
        seconds = 1
        td = 1 / samplerate

        t = []
        t = np.linspace(0, seconds, seconds * samplerate)
        if (wavet == "START"):
            xsig = np.sin(2 * 1 * np.pi * t) + np.sin(2 * 3 * np.pi * t)  # seno de 1Hz + 3Hz
        if(wavet == "SEN"):
            xsig = np.sin(2 * signal_freq * np.pi * t)
        if (wavet == "SQUARE"):
            xsig = square(2 * signal_freq * np.pi * t)
        if (wavet == "TRIANGLE"):
            xsig = sawtooth(2 * signal_freq * np.pi * t)  # seno de 1Hz + 3Hz

        Lsig = len(xsig)
        td = 1 / samplerate
        ts = 1 / samplerate2 # taxa de amostragem 50 Hz
        fator = int (ts / td)
        # envia o sinal a um amostrador
        s_out = amostragem(xsig, td, ts)

        # calcula a transformada de Fourier
        Lfft = int(pow(2, np.ceil(np.log2(Lsig) + 1)))
        Fmax = int(1 / (2 * td))
        Faxis = np.linspace(-Fmax, Fmax, int(Lfft))
        Xsig = np.fft.fftshift(np.fft.fft(xsig, Lfft))
        S_out = np.fft.fftshift(np.fft.fft(s_out, Lfft))

        # calcula o sinal reconstruído a partir de amostragem ideal e LPF (filtro passa-baixas) ideal
        # Máxima largura do LPF é igual a
        global bw_auto
        if(bw_auto):
            BW=np.floor((Lfft/fator)/2)
        else:
            BW = bw_freq_slider  # largura de banda não é maior que 10Hz

        H_lpf = np.zeros(Lfft)

        H_lpf[int(Lfft / 2 - BW):int(Lfft / 2 + BW - 1)] = 1  # LPF ideal

        S_recv = fator * S_out * H_lpf  # filtragem ideal
        s_recv = np.real(np.fft.ifft(np.fft.fftshift(S_recv)))  # domínio da freq. reconstruído
        s_recv = s_recv[0:Lsig]  # domínio do tempo reconstruçdo

    calc()

    #cria os gráficos

    TOOLTIPS = [
        ("(x,y)", "($x, $y)"),
    ]
    renderer = 'svg'
    graphTools = 'pan,wheel_zoom,box_zoom,zoom_in,zoom_out,hover,crosshair,undo,redo,reset,save'


    pltSignal = figure(plot_width=600, plot_height=300,
                        toolbar_location="below",
                        tooltips=TOOLTIPS,
                        output_backend=renderer,
                        tools=graphTools,
                        title = 'sinal g(t) e suas amostras uniformes',
                        x_axis_label = 's', y_axis_label = ''
                       )
    xsigline = pltSignal.line(t,xsig[0:Lsig], color='black', line_width=2)
    s_outline = pltSignal.line(t,s_out[0:Lsig], color='green', line_width=2)


    pltSignalFFT = figure(plot_width=600, plot_height=300,
                        toolbar_location="below",
                        tooltips=TOOLTIPS,
                        output_backend=renderer,
                        tools=graphTools,
                        title = 'Espectro de g(t)',
                        x_axis_label = 'frequência (Hz)', y_axis_label = ''
                       )
    # pltSignalFFT.x_range = Range1d(-150,150)
    # pltSignalFFT.y_range = Range1d(0,300/fator)
    Xsigline = pltSignalFFT.line(Faxis,abs(Xsig), color='black', line_width=2)
    S_outline = pltSignalFFT.line(Faxis,abs(S_out), color='green', line_width=2)


    pltSignalFFTrec = figure(plot_width=600, plot_height=300,
                        toolbar_location="below",
                        tooltips=TOOLTIPS,
                        output_backend=renderer,
                        tools=graphTools,
                        title = 'Espectro de filtragem ideal (reconstrucao)',
                        x_axis_label = 'frequência (Hz)', y_axis_label = ''
                       )

    Srecvline = pltSignalFFTrec.line(Faxis,abs(S_recv))
    # pltSignalFFTrec.x_range = Range1d(-150,150)
    # pltSignalFFTrec.y_range = Range1d(0,300)

    pltSignalrec = figure(plot_width=600, plot_height=300,
                        toolbar_location="below",
                        tooltips=TOOLTIPS,
                        output_backend=renderer,
                        tools=graphTools,
                        title = 'Sinal original versus Sinal reconstruido idealmente',
                        x_axis_label = 's', y_axis_label = ''
                       )
    xsigline2 = pltSignalrec.line(t,xsig[0:Lsig], color='black', line_width=2, line_dash = 'dashed')
    srecvline = pltSignalrec.line(t,s_recv[0:Lsig], color='green', line_width=2)

    def update(): #atualiza os valores dos gráficos
        calc()
        freq_bw_slider.update(value = BW)
        xsigline.data_source.data.update(x= t,y = xsig[0:Lsig])
        s_outline.data_source.data.update(x= t, y = s_out[0:Lsig])
        Xsigline.data_source.data.update(x= Faxis, y = abs(Xsig))
        S_outline.data_source.data.update(x= Faxis, y = abs(S_out))
        Srecvline.data_source.data.update(x= Faxis, y = abs(S_recv))
        xsigline2.data_source.data.update(x= t, y = xsig[0:Lsig])
        srecvline.data_source.data.update(x= t, y = s_recv[0:Lsig])


    def sample(attrname, old, new):
        global samplerate2
        samplerate2 = new
        update()

    def sli1(attrname, old, new):
        global signal_freq
        signal_freq = new
        update()

    def sli2(attrname, old, new):
        global bw_freq_slider
        bw_freq_slider = new
        update()

    freq_sample_slider = Slider(start=1, end=500, value=50, step=1, title="Sample Frequency")
    freq_signal_slider = Slider(start=1, end=500, value=1, step=1, title="Signal Frequency")
    freq_bw_slider = Slider(start=1, end=500, value=10, step=1, title="BW Frequency")

    freq_sample_slider.on_change("value", sample)
    freq_signal_slider.on_change("value", sli1)
    freq_bw_slider.on_change("value", sli2)

    def wavetype(event):
        #callback para o menu de seleção de onda
        global wavet
        wavet = event.item
        update()

    menu = ["START", "SEN", "SQUARE", "TRIANGLE"]
    dropdown = Dropdown(label="Type of wave", menu=menu, width_policy='min')
    dropdown.on_click(wavetype)

    def bw_callback(event):
        #callback para botão da bw automática
        global bw_auto
        bw_auto = not bw_auto
        freq_bw_slider.update(disabled=bw_auto)
        update()

    def plot_audio_bt(event):
        #callback para botão do áudio
        plot_audio()
        update()

    btn_BW = Button(label="BW AUTO", name="1", width_policy='min', button_type = 'primary')
    btn_BW.on_click(bw_callback)
    btn_audio = Button(label="GET AUDIO", name="1", width_policy='min', button_type = 'warning')
    btn_audio.on_click(plot_audio_bt)

    #Monta o layout:
    sliders = row(dropdown, freq_sample_slider, freq_signal_slider, btn_BW, freq_bw_slider, btn_audio)
    grid = gridplot([[pltSignal, pltSignalFFTrec], [pltSignalFFT, pltSignalrec]], toolbar_location = 'right', merge_tools = True)
    lay = layout(column(sliders,grid))
    doc.add_root(lay)


#roda o servidor Bokeh para funcionamento da aplicação
server = Server(
    {'/': tpiishow},
    port = 8000
)
server.start()
server.io_loop.add_callback(server.show, "/")
server.io_loop.start()
