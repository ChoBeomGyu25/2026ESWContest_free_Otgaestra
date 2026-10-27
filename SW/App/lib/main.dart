import 'package:flutter/material.dart';
import 'package:web_socket_channel/web_socket_channel.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'dart:async';
import 'dart:math' as math;
import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:google_generative_ai/google_generative_ai.dart';
import 'package:geolocator/geolocator.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const RobotArmApp());
}

// ==========================================================
// 🌐 글로벌 다국어 번역 사전 (전역 변수)
// ==========================================================
final Map<String, Map<String, String>> globalLang = {
  'ko': {
    'appTitle': '옷개스트라',
    'online': '온라인',
    'offline': '오프라인',
    'sysStart': '작업 시작',
    'sysStop': '시스템 종료',
    'setQty': '작업 수량 설정',
    'foldCount': '{count}벌 접기',
    'jobStart': '작업 시작',
    'sysOffMsg': '젯슨과 연결 후 작업 시작 버튼을 눌러주세요.',
    'sysReadyMsg': '준비 완료!\n옷을 올리고 작업을 시작해보세요.',
    'machineName': '세탁물 접기 기기',
    'autoMode': '자동 모드',
    'foldingNow': '옷 접기 중',
    'timeLeft': '남은 시간',
    'pausedMsg': '일시 정지 상태입니다.',
    'foldingMsg': '빠르고 깔끔하게 접고 있습니다',
    'progress': '진행률',
    'nextFinal': '다음 단계: 최종 정리',
    'nextFold': '다음 단계: 다음 옷 접기',
    'currentProgress': '현재 {total}벌 중 {current}번째 진행 중',
    'resume': '작업 재개하기',
    'pause': '잠시 멈춤',
    'statsBtn': '통계 데이터 보기',
    'statsTitle': '데이터 인사이트',
    'close': '닫기',
    'foldedLabel': '로봇이 갠 옷',
    'savedLabel': '아껴준 시간',
    'recentLogs': '최근 작업 로그',
    'noLogs': '작업 기록이 없습니다.',
    'logDone': '{count}벌 완료',
    'h': '시간',
    'm': '분',
    's': '초',
    'stopTitle': '⚠️ 시스템 종료',
    'stopBody': '정말로 로봇팔 시스템을 종료하시겠습니까?\n진행 중인 모든 카운트가 초기화됩니다.',
    'cancel': '취소',
    'stopBtn': '종료하기',
    'doneTitle': '작업 완료',
    'doneBody1': '선택하신 {count}벌의 옷을 모두 갰습니다.',
    'doneBody2': '💡 약 {time}를 아껴드렸습니다!',
    'ok': '확인',
    'fullTitle': '적재함 가득 참!',
    'fullBody': '옷 바구니가 가득 차서 일시 정지되었습니다.\n옷을 비운 뒤 [계속하기]를 눌러주세요.',
    'continueBtn': '계속하기',
    'errTitle': '긴급 비상 정지!',
    'netTitle': '네트워크 설정',
    'netLabel': '로봇팔 IP 및 포트',
    'save': '저장',
    'msgDisconnect': '로봇팔과 연결이 끊어졌습니다.',
    'msgNoDevice': '기기와 연결되어 있지 않습니다.',
    't1Intro': '안녕! 난 너의 옷을 예쁘게 개어줄\n꼬마 로봇 \'저비(Jeoby)\'야! 🤖',
    't1Sub': '귀찮고 둥글둥글 굴러다니는 빨래들,\n이제 나한테 싹 다 맡기고 넌 편하게 쉬어!',
    't1Next': '다음으로',
    't2Net': '기기 연결 상태를 확인하고\nIP를 설정할 수 있어요!',
    't2MainTitle': '메인 컨트롤',
    't2MainDesc': '이곳에서 옷 수량을 정하고\nJetson에 작업 시작 명령을 보내세요.',
    't2Stat': '저비가 아껴준 시간과 갠 옷의\n통계를 한눈에 볼 수 있어요!',
    't2DontShow': '앞으로 이 창을 열지 않습니다.',
    'menuTitle': '메뉴',
    'menuCody': '저비의 코디룸',
    'menuTutorial': '저비의 안내 다시 보기',
    'codyIntro1': '안녕! 지금 네가 있는 곳의\n날씨와 온도를 확인해볼게! 📡',
    'codyIntro2': '오늘 날씨에 딱 맞는\n멋진 옷차림을 추천해줄 테니 기대해! 👕',
    'codyStart': '드레스룸 열기',
    'weatherSunny': '맑음',
    'weatherRainy': '비',
    'weatherSnowy': '눈',
    'recomSunny': '오늘은 화창하고 맑아! ☀️\n가벼운 반팔이나 얇고 통풍이 잘 되는 셔츠를 추천할게!',
    'recomRainy': '밖에 비가 내리고 있어! ☔️\n물에 젖지 않게 방수 재킷과 활동하기 편한 바지를 입자!',
    'recomSnowy': '앗! 눈이 펑펑 오고 무척 춥다! ❄️\n감기 걸리지 않게 따뜻한 패딩과 목도리로 꽁꽁 싸매자!',
  },
  'en': {
    'appTitle': 'Clotheschestra',
    'online': 'Online',
    'offline': 'Offline',
    'sysStart': 'Start Folding',
    'sysStop': 'Stop System',
    'setQty': 'Set Quantity',
    'foldCount': 'Fold {count} items',
    'jobStart': 'Start Folding',
    'sysOffMsg': 'Connect to Jetson, then start folding.',
    'sysReadyMsg': 'Ready!\nPlace the clothes and start folding.',
    'machineName': 'Laundry Folding Machine',
    'autoMode': 'Auto Mode',
    'foldingNow': 'Folding...',
    'timeLeft': 'Time Left',
    'pausedMsg': 'Job is paused.',
    'foldingMsg': 'Folding quickly and neatly!',
    'progress': 'Progress',
    'nextFinal': 'Next: Final Cleanup',
    'nextFold': 'Next: Fold next item',
    'currentProgress': 'Item {current} of {total} in progress',
    'resume': 'Resume',
    'pause': 'Pause',
    'statsBtn': 'View Statistics',
    'statsTitle': 'Data Insights',
    'close': 'Close',
    'foldedLabel': 'Clothes Folded',
    'savedLabel': 'Time Saved',
    'recentLogs': 'Recent Logs',
    'noLogs': 'No records found.',
    'logDone': '{count} folded',
    'h': 'h',
    'm': 'm',
    's': 's',
    'stopTitle': '⚠️ Stop System',
    'stopBody':
        'Are you sure you want to stop?\nAll current progress will be reset.',
    'cancel': 'Cancel',
    'stopBtn': 'Stop',
    'doneTitle': 'Job Done',
    'doneBody1': 'Successfully folded {count} items.',
    'doneBody2': '💡 Saved you about {time}!',
    'ok': 'OK',
    'fullTitle': 'Basket Full!',
    'fullBody':
        'Paused because the basket is full.\nPlease empty it and press [Resume].',
    'continueBtn': 'Resume',
    'errTitle': 'Emergency Stop!',
    'netTitle': 'Network Settings',
    'netLabel': 'Robot IP & Port',
    'save': 'Save',
    'msgDisconnect': 'Disconnected from the robot.',
    'msgNoDevice': 'Device is not connected.',
    't1Intro':
        'Hi! I\'m Jeoby, your little robot\nready to fold your clothes! 🤖',
    't1Sub': 'Leave the annoying laundry to me\nand take a good rest!',
    't1Next': 'Next',
    't2Net': 'Check connection status and\nset up your IP here!',
    't2MainTitle': 'Main Control',
    't2MainDesc': 'Wake up Jeoby and select\nhow many clothes to fold.',
    't2Stat': 'View the time Jeoby saved you\nand total clothes folded!',
    't2DontShow': 'Do not show this again.',
    'menuTitle': 'Menu',
    'menuCody': 'Jeoby\'s Cody (Weather)',
    'menuTutorial': 'Show Tutorial Again',
    'codyIntro1': 'Hi! Let me check the weather\nand temperature outside! 📡',
    'codyIntro2':
        'I will recommend the perfect\noutfit for today! Stay tuned! 👕',
    'codyStart': 'Open Dress Room',
    'weatherSunny': 'Sunny',
    'weatherRainy': 'Rainy',
    'weatherSnowy': 'Snowy',
    'recomSunny':
        'It\'s sunny today! ☀️\nI recommend a light t-shirt or a breathable shirt!',
    'recomRainy':
        'It\'s raining outside! ☔️\nWear a waterproof jacket and comfortable pants!',
    'recomSnowy':
        'It\'s snowing and freezing! ❄️\nDon\'t forget a warm padded jacket and a scarf!',
  },
};

// 📝 전역 텍스트 번역 함수
String getTrans(String langCode, String key, [Map<String, String>? params]) {
  String text = globalLang[langCode]?[key] ?? globalLang['ko']![key] ?? key;
  if (params != null) {
    params.forEach((k, v) {
      text = text.replaceAll('{$k}', v);
    });
  }
  return text;
}

class RobotArmApp extends StatelessWidget {
  const RobotArmApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF2563EB)),
        scaffoldBackgroundColor: const Color(0xFFF4F5F9),
        useMaterial3: true,
      ),
      home: const ControlScreen(),
    );
  }
}

// A separate route-owned widget ensures that TextFields and their controllers
// are disposed together when the settings route actually leaves the tree.
class _NetworkSettingsResult {
  const _NetworkSettingsResult(this.target, this.token);
  final String target;
  final String token;
}

class _NetworkSettingsDialog extends StatefulWidget {
  const _NetworkSettingsDialog({
    required this.initialIp,
    required this.initialToken,
    required this.title,
    required this.ipLabel,
    required this.cancelLabel,
    required this.saveLabel,
  });

  final String initialIp;
  final String initialToken;
  final String title;
  final String ipLabel;
  final String cancelLabel;
  final String saveLabel;

  @override
  State<_NetworkSettingsDialog> createState() => _NetworkSettingsDialogState();
}

class _NetworkSettingsDialogState extends State<_NetworkSettingsDialog> {
  late final TextEditingController _ipController;
  late final TextEditingController _tokenController;
  String? _validationError;

  @override
  void initState() {
    super.initState();
    _ipController = TextEditingController(text: widget.initialIp);
    _tokenController = TextEditingController(text: widget.initialToken);
  }

  @override
  void dispose() {
    _ipController.dispose();
    _tokenController.dispose();
    super.dispose();
  }

  void _submit() {
    final input = _ipController.text.trim();
    final token = _tokenController.text.trim();
    try {
      final uri = Uri.tryParse(
        input.startsWith('ws://') || input.startsWith('wss://')
            ? input
            : 'ws://$input',
      );
      if (uri == null ||
          uri.host.isEmpty ||
          uri.userInfo.isNotEmpty ||
          !uri.hasPort ||
          uri.port < 1 ||
          uri.port > 65535 ||
          (uri.path.isNotEmpty && uri.path != '/') ||
          uri.hasQuery ||
          uri.hasFragment ||
          (uri.scheme != 'ws' && uri.scheme != 'wss') ||
          token.isEmpty) {
        setState(() => _validationError = 'IP:PORT와 Jetson 통신 키를 확인해 주세요.');
        return;
      }
      final normalized = '${uri.scheme}://${uri.authority}';
      Navigator.of(context).pop(_NetworkSettingsResult(normalized, token));
    } catch (_) {
      setState(() => _validationError = 'IP:PORT 형식을 확인해 주세요.');
    }
  }

  @override
  Widget build(BuildContext context) => AlertDialog(
    title: Text(widget.title),
    content: SingleChildScrollView(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          TextField(
            controller: _ipController,
            keyboardType: TextInputType.url,
            decoration: InputDecoration(
              labelText: widget.ipLabel,
              hintText: '192.168.107.142:8000',
            ),
          ),
          const SizedBox(height: 14),
          TextField(
            controller: _tokenController,
            obscureText: true,
            decoration: const InputDecoration(
              labelText: 'Jetson 통신 키',
              hintText: '서버 ROBOT_TOKEN과 동일하게 입력',
            ),
          ),
          if (_validationError != null) ...[
            const SizedBox(height: 10),
            Text(_validationError!, style: const TextStyle(color: Colors.red)),
          ],
          const SizedBox(height: 8),
          const Text(
            '로컬 Wi-Fi 시험용 설정입니다. 외부 인터넷에 직접 공개하지 마세요.',
            style: TextStyle(fontSize: 12, color: Colors.grey),
          ),
        ],
      ),
    ),
    actions: [
      TextButton(
        onPressed: () => Navigator.of(context).pop(),
        child: Text(widget.cancelLabel),
      ),
      ElevatedButton(onPressed: _submit, child: Text(widget.saveLabel)),
    ],
  );
}

class ControlScreen extends StatefulWidget {
  const ControlScreen({super.key});

  @override
  State<ControlScreen> createState() => _ControlScreenState();
}

// Jetson is the single source of truth for robot state and completed clothing.
// No fake progress, fake countdown or local auto-completion timers exist here.
class _ControlScreenState extends State<ControlScreen>
    with WidgetsBindingObserver {
  String currentLang = 'ko';
  bool _showTutorial = false;
  int _tutorialStep = 0;
  bool _dontShowAgain = false;

  bool isConnected = false;
  bool _hasRobotStatus = false;
  bool _connecting = false;
  bool _commandPending = false;
  bool _completionProcessing = false;
  bool _errorDialogOpen = false;
  String _backendMode = 'UNKNOWN';
  String _jobState = 'IDLE';
  String _statusMessage = '';
  String _connectionDiagnostic = '통신 서버 확인 중';
  int _retryAttempt = 0;
  bool _badCredentials = false;
  String? _activeJobId;
  int totalClothes = 1;
  int currentCount = 0;
  int _activeTarget = 1;
  int lifetimeFolded = 0;
  int lifetimeWorkSeconds = 0;
  int lastSessionWorkSeconds = 0;
  int _lastCompletedCount = 0;
  final Set<String> _loggedJobIds = <String>{};
  List<String> foldingLogs = <String>[];

  String targetIp = '';
  String _robotToken = '';
  WebSocketChannel? channel;
  StreamSubscription<dynamic>? _wsSubscription;
  Timer? _reconnectTimer;
  Timer? _heartbeatTimer;
  Timer? _authenticationTimer;
  int _connectionEpoch = 0;
  DateTime _lastPong = DateTime.now();

  static const Color clrTextMain = Color(0xFF1E293B);
  static const Color clrTextSub = Color(0xFF64748B);
  static const Color clrMainBlue = Color(0xFF2563EB);
  static const Color clrDanger = Color(0xFFEF4444);
  static const Color clrWarning = Color(0xFFF59E0B);
  static const Color clrSuccess = Color(0xFF10B981);

  String t(String key, [Map<String, String>? params]) =>
      getTrans(currentLang, key, params);

  bool get isFolding => _jobState == 'RUNNING' || _jobState == 'STOP_REQUESTED';
  bool get _canControl =>
      isConnected &&
      _hasRobotStatus &&
      (_backendMode == 'DEMO_ONLY' || _backendMode == 'PHYSICAL');
  bool get _isDemo => _backendMode == 'DEMO_ONLY';
  String get _connectionLabel {
    if (!isConnected) return currentLang == 'en' ? 'Disconnected' : '연결 안 됨';
    if (!_hasRobotStatus)
      return currentLang == 'en' ? 'Synchronizing' : '동기화 중';
    return _isDemo ? 'TEST MODE' : t('online');
  }

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initializeApp();
  }

  Future<void> _initializeApp() async {
    await _loadSettingsAndStats();
    if (mounted) _connectWebSocket();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.resumed) {
      // The mobile OS may suspend timers and sockets while in the background.
      _lastPong = DateTime.now();
      if (isConnected) {
        _sendData({'status': 'GET_STATUS'}, quietly: true);
        _sendData({'status': 'PING'}, quietly: true);
      } else if (!_connecting) {
        _reconnectTimer?.cancel();
        _connectWebSocket();
      }
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _connectionEpoch++;
    _reconnectTimer?.cancel();
    _heartbeatTimer?.cancel();
    _authenticationTimer?.cancel();
    _wsSubscription?.cancel();
    channel?.sink.close();
    super.dispose();
  }

  Future<void> _loadSettingsAndStats() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      if (!mounted) return;
      final previousLogs = prefs.getStringList('foldingLogs') ?? <String>[];
      setState(() {
        targetIp = prefs.getString('robotTargetIp') ?? '';
        _robotToken = prefs.getString('robotLabToken') ?? '';
        currentLang = prefs.getString('appLanguage') ?? 'ko';
        _showTutorial = !(prefs.getBool('hasSeenTutorial') ?? false);
        lifetimeFolded = prefs.getInt('savedFolded') ?? 0;
        // savedTime was a guessed savings formula in the previous app.
        // Do not silently relabel that estimate as measured robot run time.
        lifetimeWorkSeconds = prefs.getInt('savedRobotWorkSeconds') ?? 0;
        foldingLogs = previousLogs;
        _loggedJobIds.addAll(prefs.getStringList('loggedJobIds') ?? <String>[]);
      });
      for (final raw in previousLogs) {
        try {
          final value = jsonDecode(raw);
          if (value is Map && value['jobId'] is String) {
            _loggedJobIds.add(value['jobId'] as String);
          }
        } catch (_) {}
      }
      final oldCount = foldingLogs.length;
      _cleanUpOldLogs();
      if (oldCount != foldingLogs.length) await _saveStatisticsData();
    } catch (e) {
      debugPrint('[APP] saved settings error: $e');
    }
  }

  Future<void> _saveStatisticsData() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.setStringList('foldingLogs', foldingLogs);
      await prefs.setStringList(
        'loggedJobIds',
        _loggedJobIds.toList().reversed.take(300).toList(),
      );
      await prefs.setInt('savedFolded', lifetimeFolded);
      await prefs.setInt('savedRobotWorkSeconds', lifetimeWorkSeconds);
    } catch (e) {
      debugPrint('[APP] save statistics error: $e');
    }
  }

  void _cleanUpOldLogs() {
    final cutoff = DateTime.now().subtract(const Duration(days: 7));
    foldingLogs.retainWhere((raw) {
      try {
        final map = jsonDecode(raw);
        if (map is! Map) return false;
        final iso = map['recordedAt'];
        DateTime? date = iso is String ? DateTime.tryParse(iso) : null;
        // Historical logs have the form YYYY.MM.DD HH:MM.
        if (date == null && map['time'] is String) {
          date = DateTime.tryParse(
            (map['time'] as String).replaceAll('.', '-') + ':00',
          );
        }
        return date != null && date.isAfter(cutoff);
      } catch (_) {
        return false;
      }
    });
  }

  Future<void> _changeLanguage(String language) async {
    setState(() => currentLang = language);
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('appLanguage', language);
  }

  Future<void> _closeTutorial() async {
    if (_dontShowAgain) {
      final prefs = await SharedPreferences.getInstance();
      await prefs.setBool('hasSeenTutorial', true);
    }
    if (mounted) setState(() => _showTutorial = false);
  }

  Uri _serverUri() {
    final input = targetIp.trim();
    final uri = Uri.parse(
      input.startsWith('ws://') || input.startsWith('wss://')
          ? input
          : 'ws://$input',
    );
    return uri.replace(path: '/ws', query: null, fragment: null);
  }

  Future<void> _connectWebSocket() async {
    if (!mounted || _connecting || isConnected || _badCredentials) return;
    _reconnectTimer?.cancel();
    if (_robotToken.trim().isEmpty) {
      setState(() => _connectionDiagnostic = '네트워크 설정에서 Jetson 통신 키를 입력하세요.');
      return;
    }
    _connecting = true;
    setState(() => _connectionDiagnostic = 'Jetson 연결 시도 중 ($targetIp)');
    final epoch = ++_connectionEpoch;
    try {
      final candidate = WebSocketChannel.connect(_serverUri());
      channel = candidate;
      _wsSubscription = candidate.stream.listen(
        (message) {
          if (mounted && epoch == _connectionEpoch) _receiveMessage(message);
        },
        onDone: () => _disconnect(epoch, reason: 'Jetson 통신 서버가 연결을 종료했습니다.'),
        onError: (Object error) {
          debugPrint('[WS] socket error: $error');
          _disconnect(epoch, reason: 'WebSocket 연결 오류. IP와 서버 실행 여부를 확인하세요.');
        },
        cancelOnError: true,
      );
      await candidate.ready.timeout(const Duration(seconds: 6));
      if (!mounted || epoch != _connectionEpoch) return;
      _connecting = false;
      // Authenticate before accepting robot state or commands.
      candidate.sink.add(jsonEncode({'status': 'AUTH', 'token': _robotToken}));
      _authenticationTimer?.cancel();
      _authenticationTimer = Timer(const Duration(seconds: 6), () {
        if (epoch == _connectionEpoch && !isConnected) {
          _disconnect(epoch, reason: '연결했지만 6초 안에 인증 응답을 받지 못했습니다.');
        }
      });
    } catch (e) {
      debugPrint('[WS] connect failed (${_serverUri()}): $e');
      if (epoch == _connectionEpoch) {
        _disconnect(epoch, reason: '서버 연결 실패. Jetson의 8000 포트와 Wi-Fi를 확인하세요.');
      }
    }
  }

  void _disconnect(int epoch, {String? reason}) {
    if (!mounted || epoch != _connectionEpoch) return;
    ++_connectionEpoch; // Ignore onDone from an older socket.
    final previouslyConnected = isConnected;
    _connecting = false;
    _commandPending = false;
    _authenticationTimer?.cancel();
    _heartbeatTimer?.cancel();
    _wsSubscription?.cancel();
    _wsSubscription = null;
    channel?.sink.close();
    channel = null;
    setState(() {
      isConnected = false;
      _hasRobotStatus = false;
      _backendMode = 'UNKNOWN';
      // Keep last robot state visible: losing Wi-Fi does NOT stop the arms.
      _statusMessage = '통신이 끊겼습니다. 실제 로봇 상태를 현장에서 확인하세요.';
      _connectionDiagnostic = reason ?? '연결이 끊어졌습니다. 재연결을 시도합니다.';
    });
    if (previouslyConnected) _showSnackBar(t('msgDisconnect'), isError: true);
    _reconnectTimer?.cancel();
    // Limit repeated attempts when Jetson is offline; keep Flutter UI responsive.
    const retrySeconds = [3, 5, 10, 20, 30];
    final delay =
        retrySeconds[_retryAttempt.clamp(0, retrySeconds.length - 1).toInt()];
    if (_retryAttempt < retrySeconds.length - 1) _retryAttempt++;
    _reconnectTimer = Timer(Duration(seconds: delay), _connectWebSocket);
  }

  void _restartWebSocket() {
    _badCredentials = false;
    _retryAttempt = 0;
    ++_connectionEpoch;
    _connecting = false;
    _reconnectTimer?.cancel();
    _authenticationTimer?.cancel();
    _heartbeatTimer?.cancel();
    _wsSubscription?.cancel();
    _wsSubscription = null;
    channel?.sink.close();
    channel = null;
    setState(() {
      isConnected = false;
      _hasRobotStatus = false;
      _backendMode = 'UNKNOWN';
    });
    _connectWebSocket();
  }

  void _beginHeartbeat() {
    _heartbeatTimer?.cancel();
    _lastPong = DateTime.now();
    _heartbeatTimer = Timer.periodic(const Duration(seconds: 10), (_) {
      if (!mounted || !isConnected) return;
      if (DateTime.now().difference(_lastPong) > const Duration(seconds: 35)) {
        _disconnect(_connectionEpoch);
        return;
      }
      _sendData({'status': 'PING'}, quietly: true);
    });
  }

  void _sendData(Map<String, dynamic> payload, {bool quietly = false}) {
    if (!isConnected || channel == null) {
      if (!quietly) _showSnackBar(t('msgNoDevice'), isError: true);
      return;
    }
    try {
      channel!.sink.add(jsonEncode(payload));
    } catch (e) {
      debugPrint('[WS] send error: $e');
      _disconnect(_connectionEpoch);
    }
  }

  void _receiveMessage(dynamic message) {
    try {
      final decoded = jsonDecode(message as String);
      if (decoded is! Map<String, dynamic> || !mounted) return;
      final data = decoded;
      switch (data['type']) {
        case 'AUTH_OK':
          _authenticationTimer?.cancel();
          _retryAttempt = 0;
          setState(() {
            isConnected = true;
            _connectionDiagnostic = '';
          });
          _lastPong = DateTime.now();
          _beginHeartbeat();
          _sendData({'status': 'GET_STATUS'});
          return;
        case 'AUTH_FAILED':
          _showSnackBar('인증 실패: 설정에서 Jetson 통신 키를 확인하세요.', isError: true);
          // Intentional disconnect; do not retry bad credentials automatically.
          _badCredentials = true;
          _connectionEpoch++;
          _heartbeatTimer?.cancel();
          _authenticationTimer?.cancel();
          _wsSubscription?.cancel();
          channel?.sink.close();
          channel = null;
          _connecting = false;
          if (mounted)
            setState(() {
              isConnected = false;
              _hasRobotStatus = false;
              _connectionDiagnostic = 'Jetson이 통신 키를 거부했습니다. 설정에서 키를 확인하세요.';
            });
          return;
        case 'PONG':
          _lastPong = DateTime.now();
          return;
        case 'STATUS_UPDATE':
          _commandPending = false;
          _lastPong = DateTime.now();
          setState(() {
            _hasRobotStatus = true;
            _backendMode = data['source'] is String
                ? data['source'] as String
                : 'UNKNOWN';
            _jobState = data['jobState'] is String
                ? data['jobState'] as String
                : 'IDLE';
            _statusMessage = data['message'] is String
                ? data['message'] as String
                : '';
            _activeJobId = data['activeJobId'] is String
                ? data['activeJobId'] as String
                : null;
            if (data['totalTarget'] is int) {
              _activeTarget = (data['totalTarget'] as int).clamp(1, 30).toInt();
              if (isFolding) totalClothes = _activeTarget;
            }
            if (data['currentCount'] is int) {
              currentCount = (data['currentCount'] as int)
                  .clamp(0, _activeTarget)
                  .toInt();
            }
          });
          final last = data['lastCompletedJob'];
          if (last is Map<String, dynamic>) _recordCompletion(last);
          return;
        case 'JOB_COMPLETED':
          _commandPending = false;
          _recordCompletion(data);
          return;
        case 'COMMAND_REJECTED':
          setState(() => _commandPending = false);
          _showSnackBar(
            '${data['message'] ?? '명령을 수행할 수 없습니다.'}',
            isError: true,
          );
          return;
        case 'ERROR':
          setState(() => _commandPending = false);
          _showError('${data['message'] ?? 'Jetson이 오류를 보고했습니다.'}');
          return;
        default:
          debugPrint('[WS] unrecognized server payload: ${data['type']}');
      }
    } catch (e) {
      debugPrint('[WS] invalid JSON or payload: $e');
    }
  }

  // Logs are written only for a uniquely identified Jetson completion flag.
  // STATUS_UPDATE alone can never increment historical statistics.
  Future<void> _recordCompletion(Map<String, dynamic> data) async {
    final id = data['jobId'];
    final count = data['completedCount'];
    if (!mounted ||
        id is! String ||
        id.isEmpty ||
        count is! int ||
        count < 1 ||
        _loggedJobIds.contains(id) ||
        _completionProcessing)
      return;
    _completionProcessing = true;
    try {
      final seconds = data['durationSeconds'] is num
          ? (data['durationSeconds'] as num).round().clamp(0, 864000).toInt()
          : 0;
      if (_isDemo) {
        _loggedJobIds.add(id);
        final prefs = await SharedPreferences.getInstance();
        await prefs.setStringList('loggedJobIds', _loggedJobIds.toList());
        if (mounted) {
          setState(() {
            _jobState = 'IDLE';
            _lastCompletedCount = count;
          });
          _showSnackBar('통신 테스트 완료: $count벌 (통계에는 기록하지 않음)');
        }
        return;
      }
      if (_backendMode != 'PHYSICAL') return;
      final timestamp = data['completedAt'];
      final now =
          (timestamp is String
              ? DateTime.tryParse(timestamp)?.toLocal()
              : null) ??
          DateTime.now();
      final newLog = <String, dynamic>{
        'jobId': id,
        'recordedAt': now.toIso8601String(),
        'time':
            '${now.year}.${now.month.toString().padLeft(2, '0')}.${now.day.toString().padLeft(2, '0')} '
            '${now.hour.toString().padLeft(2, '0')}:${now.minute.toString().padLeft(2, '0')}',
        'count': count,
        'durationSeconds': seconds,
      };
      // Mutate once; persist before the popup, so reconnection doesn't recount.
      _loggedJobIds.add(id);
      setState(() {
        _jobState = 'IDLE';
        _activeJobId = null;
        _lastCompletedCount = count;
        lastSessionWorkSeconds = seconds;
        lifetimeFolded += count;
        lifetimeWorkSeconds += seconds;
        foldingLogs.insert(0, jsonEncode(newLog));
        _cleanUpOldLogs();
      });
      await _saveStatisticsData();
      if (mounted) {
        WidgetsBinding.instance.addPostFrameCallback((_) {
          if (mounted) _showCompletionPopup(count, seconds);
        });
      }
    } finally {
      _completionProcessing = false;
    }
  }

  void startFolding() {
    if (!_canControl || isFolding || _commandPending) return;
    setState(() => _commandPending = true);
    _sendData({'status': 'START_FOLDING', 'total_target': totalClothes});
    // A command is a request. Only Jetson STATUS_UPDATE means arms are running.
  }

  Future<void> requestStop() async {
    if (!_canControl || !isFolding || _commandPending) return;
    final shouldStop = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('작업 중지 요청'),
        content: const Text(
          '진행 중인 작업을 Jetson에 중지 요청할까요? '
          '이 버튼은 물리 비상정지 장치를 대신하지 않습니다.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx, false),
            child: Text(t('cancel')),
          ),
          TextButton(
            onPressed: () => Navigator.pop(ctx, true),
            child: Text(t('stopBtn')),
          ),
        ],
      ),
    );
    if (shouldStop != true || !mounted) return;
    setState(() => _commandPending = true);
    _sendData({'status': 'STOP_JOB'});
  }

  void _showError(String message) {
    if (!mounted || _errorDialogOpen) return;
    _errorDialogOpen = true;
    showDialog<void>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: Text(t('errTitle')),
        content: Text(message),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx), child: Text(t('ok'))),
        ],
      ),
    ).whenComplete(() => _errorDialogOpen = false);
  }

  Future<void> _showIpSettingsDialog() async {
    // Keep TextEditingControllers inside the dialog State.  Disposing them
    // from showDialog().whenComplete can race the route's reverse animation.
    final result = await showDialog<_NetworkSettingsResult>(
      context: context,
      builder: (_) => _NetworkSettingsDialog(
        initialIp: targetIp,
        initialToken: _robotToken,
        title: t('netTitle'),
        ipLabel: t('netLabel'),
        cancelLabel: t('cancel'),
        saveLabel: t('save'),
      ),
    );
    if (!mounted || result == null) return;

    // Do not rebuild/reconnect while the settings route is being torn down.
    // The dialog owns and disposes its text controllers in its own dispose().
    await Future<void>.delayed(const Duration(milliseconds: 350));
    if (!mounted) return;
    setState(() {
      targetIp = result.target;
      _robotToken = result.token;
      _connectionDiagnostic = '변경된 Jetson 주소로 연결 중입니다.';
    });
    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.setString('robotTargetIp', result.target);
      // For a trusted lab LAN only. Real deployments need secure storage/TLS.
      await prefs.setString('robotLabToken', result.token);
    } catch (e) {
      debugPrint('[APP] save network configuration failed: $e');
      if (mounted)
        _showSnackBar(
          '설정을 저장하지 못했습니다. 앱을 재시작하면 이전 설정으로 돌아갈 수 있어요.',
          isError: true,
        );
    }
    if (!mounted) return;
    _restartWebSocket();
  }

  void _showSnackBar(String message, {bool isError = false}) {
    if (!mounted) return;
    final messenger = ScaffoldMessenger.of(context);
    messenger.hideCurrentSnackBar();
    messenger.showSnackBar(
      SnackBar(
        content: Text(message),
        backgroundColor: isError ? clrDanger : clrMainBlue,
        behavior: SnackBarBehavior.floating,
        duration: const Duration(seconds: 3),
      ),
    );
  }

  String formatDashboardTime(int seconds) {
    final h = seconds ~/ 3600;
    final m = (seconds % 3600) ~/ 60;
    final s = seconds % 60;
    if (h > 0) return '${h}시간 ${m}분';
    if (m > 0) return '${m}분 ${s}초';
    return '${s}초';
  }

  void _showCompletionPopup(int count, int seconds) {
    showDialog<void>(
      context: context,
      builder: (ctx) => AlertDialog(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
        title: Row(
          children: [
            const Icon(Icons.check_circle, color: clrMainBlue),
            const SizedBox(width: 10),
            Text(t('doneTitle')),
          ],
        ),
        content: Text(
          '${count}벌의 접기 완료 신호를 Jetson에서 받았습니다.\n'
          '실제 작업 시간: ${formatDashboardTime(seconds)}',
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx), child: Text(t('ok'))),
        ],
      ),
    );
  }

  void showDataDashboard() {
    showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      showDragHandle: true,
      backgroundColor: Colors.white,
      builder: (sheetContext) => SafeArea(
        child: Padding(
          padding: const EdgeInsets.fromLTRB(20, 5, 20, 22),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text(
                    t('statsTitle'),
                    style: const TextStyle(
                      fontSize: 22,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  IconButton(
                    onPressed: () => Navigator.pop(sheetContext),
                    icon: const Icon(Icons.close),
                  ),
                ],
              ),
              const Divider(),
              Row(
                children: [
                  Expanded(
                    child: _buildStatCard(
                      title: t('foldedLabel'),
                      value: '$lifetimeFolded',
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: _buildStatCard(
                      title: currentLang == 'en'
                          ? 'Robot runtime'
                          : '누적 로봇 작업 시간',
                      value: formatDashboardTime(lifetimeWorkSeconds),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 20),
              Text(
                t('recentLogs'),
                style: const TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(height: 8),
              SizedBox(
                height: 240,
                child: foldingLogs.isEmpty
                    ? Center(child: Text(t('noLogs')))
                    : ListView.separated(
                        itemCount: foldingLogs.length,
                        separatorBuilder: (_, __) => const Divider(height: 1),
                        itemBuilder: (context, index) {
                          final entry =
                              jsonDecode(foldingLogs[index])
                                  as Map<String, dynamic>;
                          final count = entry['count'] ?? 0;
                          final duration = entry['durationSeconds'];
                          return ListTile(
                            contentPadding: EdgeInsets.zero,
                            title: Text(
                              t('logDone', {'count': '$count'}),
                              style: const TextStyle(
                                fontWeight: FontWeight.w600,
                              ),
                            ),
                            subtitle: duration is num
                                ? Text(
                                    '로봇 작업 시간: ${formatDashboardTime(duration.round())}',
                                  )
                                : null,
                            trailing: Text(
                              '${entry['time'] ?? ''}',
                              style: const TextStyle(fontSize: 12),
                            ),
                          );
                        },
                      ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildStatCard({required String title, required String value}) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: clrMainBlue.withOpacity(0.06),
        borderRadius: BorderRadius.circular(14),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(title, style: const TextStyle(fontSize: 12, color: clrTextSub)),
          const SizedBox(height: 5),
          Text(
            value,
            style: const TextStyle(
              fontSize: 22,
              fontWeight: FontWeight.bold,
              color: clrMainBlue,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildDrawer() {
    return Drawer(
      backgroundColor: Colors.white,
      child: Column(
        children: [
          Container(
            width: double.infinity,
            padding: const EdgeInsets.only(top: 60, bottom: 20, left: 20),
            decoration: const BoxDecoration(
              color: clrMainBlue,
              borderRadius: BorderRadius.only(bottomRight: Radius.circular(30)),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Icon(Icons.checkroom, color: Colors.white, size: 40),
                const SizedBox(height: 10),
                Text(
                  t('menuTitle'),
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 24,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
          ),
          Expanded(
            child: ListView(
              padding: const EdgeInsets.symmetric(vertical: 10),
              children: [
                ListTile(
                  leading: const Icon(
                    Icons.wb_sunny_rounded,
                    color: clrMainBlue,
                  ),
                  title: Text(
                    t('menuCody'),
                    style: const TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.bold,
                      color: clrTextMain,
                    ),
                  ),
                  onTap: () {
                    Navigator.pop(context);
                    Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (context) =>
                            JeobyCodyScreen(langCode: currentLang),
                      ),
                    );
                  },
                ),
                ListTile(
                  leading: const Icon(Icons.help_outline, color: clrMainBlue),
                  title: Text(
                    t('menuTutorial'),
                    style: const TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.bold,
                      color: clrTextMain,
                    ),
                  ),
                  onTap: () async {
                    Navigator.pop(context);
                    final prefs = await SharedPreferences.getInstance();
                    await prefs.setBool('hasSeenTutorial', false);
                    setState(() {
                      _dontShowAgain = false;
                      _tutorialStep = 0;
                      _showTutorial = true;
                    });
                  },
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  // ==========================================================
  // 🌟 튜토리얼 UI 오버레이 위젯 세트
  // ==========================================================
  Widget _buildTutorialOverlay() {
    return Positioned.fill(
      child: Material(
        color: Colors.black.withOpacity(0.75),
        child: _tutorialStep == 0
            ? SafeArea(child: _buildJeobyIntro())
            : _buildAppGuide(),
      ),
    );
  }

  Widget _buildJeobyIntro() {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            margin: const EdgeInsets.symmetric(horizontal: 40),
            padding: const EdgeInsets.all(25),
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(20),
              boxShadow: const [
                BoxShadow(
                  color: Colors.black26,
                  blurRadius: 10,
                  offset: Offset(0, 5),
                ),
              ],
            ),
            child: Column(
              children: [
                Text(
                  t('t1Intro'),
                  textAlign: TextAlign.center,
                  style: const TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                    color: clrTextMain,
                    height: 1.4,
                  ),
                ),
                const SizedBox(height: 15),
                Text(
                  t('t1Sub'),
                  textAlign: TextAlign.center,
                  style: const TextStyle(
                    fontSize: 15,
                    color: clrTextSub,
                    height: 1.4,
                  ),
                ),
              ],
            ),
          ),
          const Icon(Icons.arrow_drop_down, color: Colors.white, size: 60),
          const CuteRobot(state: RobotState.idle),
          const SizedBox(height: 50),
          ElevatedButton(
            onPressed: () {
              setState(() => _tutorialStep = 1);
            },
            style: ElevatedButton.styleFrom(
              backgroundColor: clrMainBlue,
              foregroundColor: Colors.white,
              padding: const EdgeInsets.symmetric(horizontal: 40, vertical: 15),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(30),
              ),
            ),
            child: Text(
              t('t1Next'),
              style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildHighlightedUI() {
    return Scaffold(
      backgroundColor: Colors.transparent,
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        title: Opacity(
          opacity: 0,
          child: Text(t('appTitle'), style: const TextStyle(fontSize: 22)),
        ),
        actions: [
          Opacity(
            opacity: 0,
            child: Container(
              margin: const EdgeInsets.only(right: 10),
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
              child: Row(
                children: [
                  const Icon(Icons.circle, size: 10),
                  const SizedBox(width: 6),
                  Text(t('offline')),
                ],
              ),
            ),
          ),
          Opacity(
            opacity: 0,
            child: IconButton(
              icon: const Icon(Icons.language, size: 24),
              onPressed: () {},
            ),
          ),
          Opacity(
            opacity: 0,
            child: IconButton(
              icon: const Icon(Icons.settings, size: 28),
              onPressed: () {},
            ),
          ),
        ],
      ),
      body: SingleChildScrollView(
        physics: const NeverScrollableScrollPhysics(),
        padding: const EdgeInsets.symmetric(horizontal: 24.0, vertical: 10.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            ElevatedButton(
              onPressed: () {},
              style: ElevatedButton.styleFrom(
                padding: const EdgeInsets.symmetric(vertical: 20),
                backgroundColor: clrMainBlue,
                foregroundColor: Colors.white,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(16),
                ),
                elevation: 10,
                shadowColor: Colors.black54,
              ),
              child: Text(
                t('sysStart'),
                style: const TextStyle(
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),

            Opacity(
              opacity: 0.0,
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  const SizedBox(height: 30),
                  Text(t('setQty'), style: const TextStyle(fontSize: 16)),
                  const SizedBox(height: 10),
                  Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 20,
                      vertical: 5,
                    ),
                    decoration: BoxDecoration(
                      border: Border.all(color: Colors.transparent),
                    ),
                    child: DropdownButtonHideUnderline(
                      child: DropdownButton<int>(
                        value: 1,
                        isExpanded: true,
                        items: [
                          DropdownMenuItem(
                            value: 1,
                            child: Text(t('foldCount', {'count': '1'})),
                          ),
                        ],
                        onChanged: null,
                      ),
                    ),
                  ),
                  const SizedBox(height: 25),
                  const SizedBox(height: 50),
                  Center(
                    child: Column(
                      children: [
                        const CuteRobot(state: RobotState.idle),
                        const SizedBox(height: 25),
                        Text(
                          t('sysOffMsg'),
                          style: const TextStyle(fontSize: 16, height: 1.5),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 30),
                ],
              ),
            ),

            Center(
              child: TextButton.icon(
                onPressed: () {},
                style: TextButton.styleFrom(
                  backgroundColor: Colors.white,
                  padding: const EdgeInsets.symmetric(
                    horizontal: 40,
                    vertical: 14,
                  ),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                ),
                icon: const Icon(Icons.insights, color: clrMainBlue),
                label: Text(
                  t('statsBtn'),
                  style: const TextStyle(
                    color: clrMainBlue,
                    fontWeight: FontWeight.bold,
                    fontSize: 16,
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildAppGuide() {
    return Stack(
      children: [
        _buildHighlightedUI(),
        SafeArea(
          child: Stack(
            children: [
              const Positioned(
                top: 135,
                left: 0,
                right: 0,
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.center,
                  children: [
                    Icon(
                      Icons.arrow_upward_rounded,
                      color: Colors.white,
                      size: 30,
                    ),
                  ],
                ),
              ),
              Positioned(
                top: 170,
                left: 0,
                right: 0,
                child: Column(
                  children: [
                    Text(
                      t('t2MainTitle'),
                      style: const TextStyle(
                        color: Color(0xFF60A5FA),
                        fontSize: 18,
                        fontWeight: FontWeight.w900,
                      ),
                    ),
                    const SizedBox(height: 5),
                    Text(
                      t('t2MainDesc'),
                      textAlign: TextAlign.center,
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 16,
                        height: 1.4,
                      ),
                    ),
                  ],
                ),
              ),

              // 🌟 수정된 부분: 통계 데이터 튜토리얼 글씨 위치 (230 -> 130)
              Positioned(
                bottom: 250,
                left: 0,
                right: 0,
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.center,
                  children: [
                    const Icon(
                      Icons.arrow_upward_rounded,
                      color: Colors.white,
                      size: 30,
                    ),
                    const SizedBox(height: 10),
                    Text(
                      t('t2Stat'),
                      textAlign: TextAlign.center,
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 16,
                        height: 1.4,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ],
                ),
              ),

              Align(
                alignment: Alignment.bottomCenter,
                child: Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 15,
                    vertical: 10,
                  ),
                  color: Colors.black.withOpacity(0.5),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Row(
                        children: [
                          Theme(
                            data: ThemeData(
                              unselectedWidgetColor: Colors.white,
                            ),
                            child: Checkbox(
                              value: _dontShowAgain,
                              activeColor: clrMainBlue,
                              checkColor: Colors.white,
                              onChanged: (bool? value) {
                                setState(() {
                                  _dontShowAgain = value ?? false;
                                });
                              },
                            ),
                          ),
                          GestureDetector(
                            onTap: () => setState(
                              () => _dontShowAgain = !_dontShowAgain,
                            ),
                            child: Text(
                              t('t2DontShow'),
                              style: const TextStyle(
                                color: Colors.white,
                                fontSize: 15,
                              ),
                            ),
                          ),
                        ],
                      ),
                      IconButton(
                        icon: const Icon(
                          Icons.close,
                          color: Colors.white,
                          size: 30,
                        ),
                        onPressed: _closeTutorial,
                      ),
                    ],
                  ),
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }
  // ==========================================================

  @override
  Widget build(BuildContext context) {
    final isStopping = _jobState == 'STOP_REQUESTED';
    final isError = _jobState == 'ERROR';
    final canStart = _canControl && !isFolding && !isError && !_commandPending;
    return Stack(
      children: [
        Scaffold(
          drawer: _buildDrawer(),
          appBar: AppBar(
            title: Text(
              t('appTitle'),
              style: const TextStyle(
                color: clrTextMain,
                fontSize: 22,
                fontWeight: FontWeight.w800,
              ),
            ),
            iconTheme: const IconThemeData(color: clrTextMain),
            actions: [
              Container(
                margin: const EdgeInsets.only(right: 6),
                padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 6),
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(20),
                  color: isConnected
                      ? clrSuccess.withOpacity(.1)
                      : clrDanger.withOpacity(.08),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(
                      Icons.circle,
                      size: 9,
                      color: isConnected ? clrSuccess : clrDanger,
                    ),
                    const SizedBox(width: 4),
                    Text(
                      _connectionLabel,
                      style: TextStyle(
                        fontSize: 11,
                        fontWeight: FontWeight.bold,
                        color: isConnected ? clrSuccess : clrDanger,
                      ),
                    ),
                  ],
                ),
              ),
              PopupMenuButton<String>(
                icon: const Icon(Icons.language, color: clrTextSub),
                onSelected: _changeLanguage,
                itemBuilder: (_) => const [
                  PopupMenuItem(value: 'ko', child: Text('🇰🇷 한국어')),
                  PopupMenuItem(value: 'en', child: Text('🇺🇸 English')),
                  PopupMenuItem(value: 'zh', child: Text('🇨🇳 中文')),
                  PopupMenuItem(value: 'ja', child: Text('🇯🇵 日本語')),
                  PopupMenuItem(value: 'es', child: Text('🇪🇸 Español')),
                ],
              ),
              IconButton(
                icon: const Icon(Icons.settings, color: clrTextSub),
                onPressed: _showIpSettingsDialog,
              ),
            ],
          ),
          body: SingleChildScrollView(
            padding: const EdgeInsets.fromLTRB(24, 16, 24, 40),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                if (!_canControl) ...[
                  Container(
                    padding: const EdgeInsets.all(13),
                    decoration: BoxDecoration(
                      color: clrWarning.withOpacity(.12),
                      borderRadius: BorderRadius.circular(14),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          _robotToken.isEmpty
                              ? '통신 설정에서 IP와 Jetson 통신 키를 입력하세요.'
                              : _connectionDiagnostic,
                          style: const TextStyle(
                            fontSize: 13,
                            color: clrTextMain,
                          ),
                        ),
                        Text(
                          '설정 주소: $targetIp',
                          style: const TextStyle(
                            fontSize: 12,
                            color: clrTextSub,
                          ),
                        ),
                        Wrap(
                          spacing: 8,
                          children: [
                            TextButton.icon(
                              onPressed: _showIpSettingsDialog,
                              icon: const Icon(Icons.wifi),
                              label: Text(t('netTitle')),
                            ),
                            TextButton.icon(
                              onPressed: _robotToken.isEmpty
                                  ? _showIpSettingsDialog
                                  : _restartWebSocket,
                              icon: const Icon(Icons.refresh),
                              label: const Text('지금 재연결'),
                            ),
                          ],
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 18),
                ],
                if (_isDemo) ...[
                  Container(
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      borderRadius: BorderRadius.circular(12),
                      color: clrWarning.withOpacity(.12),
                    ),
                    child: const Text(
                      '통신 시험 모드: 로봇팔은 실제로 움직이지 않고 통계도 저장하지 않습니다.',
                      style: TextStyle(fontSize: 12, color: clrTextMain),
                    ),
                  ),
                  const SizedBox(height: 14),
                ],
                if (_statusMessage.isNotEmpty &&
                    (isError || !isConnected || isStopping)) ...[
                  Container(
                    padding: const EdgeInsets.all(13),
                    decoration: BoxDecoration(
                      borderRadius: BorderRadius.circular(12),
                      color: (isError ? clrDanger : clrWarning).withOpacity(.1),
                    ),
                    child: Text(
                      _statusMessage,
                      style: const TextStyle(fontSize: 13),
                    ),
                  ),
                  const SizedBox(height: 14),
                ],
                if (!isFolding) ...[
                  Text(
                    t('setQty'),
                    style: const TextStyle(
                      fontSize: 16,
                      color: clrTextMain,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 10),
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 16),
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(16),
                      border: Border.all(color: Colors.black12),
                    ),
                    child: DropdownButtonHideUnderline(
                      child: DropdownButton<int>(
                        isExpanded: true,
                        value: totalClothes,
                        items: List.generate(
                          30,
                          (index) => DropdownMenuItem(
                            value: index + 1,
                            child: Text(
                              t('foldCount', {'count': '${index + 1}'}),
                            ),
                          ),
                        ),
                        onChanged: canStart
                            ? (value) {
                                if (value != null)
                                  setState(() => totalClothes = value);
                              }
                            : null,
                      ),
                    ),
                  ),
                  const SizedBox(height: 22),
                  ElevatedButton.icon(
                    onPressed: canStart ? startFolding : null,
                    icon: const Icon(Icons.play_arrow_rounded, size: 25),
                    label: Padding(
                      padding: const EdgeInsets.symmetric(vertical: 18),
                      child: Text(
                        _commandPending ? 'Jetson 응답 대기 중' : t('jobStart'),
                        style: const TextStyle(
                          fontSize: 19,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: clrMainBlue,
                      foregroundColor: Colors.white,
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(16),
                      ),
                    ),
                  ),
                  const SizedBox(height: 36),
                  CuteRobot(
                    state: isError ? RobotState.paused : RobotState.idle,
                  ),
                  const SizedBox(height: 16),
                  Text(
                    isError
                        ? 'Jetson 오류를 현장에서 확인한 뒤 복구하세요.'
                        : _canControl
                        ? '준비 완료! 수량을 선택하고 작업을 시작하세요.'
                        : 'Jetson과 연결하면 작업을 시작할 수 있어요.',
                    textAlign: TextAlign.center,
                    style: const TextStyle(color: clrTextSub, fontSize: 15),
                  ),
                ] else ...[
                  Card(
                    elevation: 0,
                    color: Colors.white,
                    shape: RoundedRectangleBorder(
                      side: const BorderSide(color: Colors.black12),
                      borderRadius: BorderRadius.circular(20),
                    ),
                    child: Padding(
                      padding: const EdgeInsets.all(23),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Row(
                            mainAxisAlignment: MainAxisAlignment.spaceBetween,
                            children: [
                              Text(
                                t('machineName'),
                                style: const TextStyle(
                                  color: clrTextSub,
                                  fontWeight: FontWeight.w600,
                                ),
                              ),
                              const Text(
                                '자동 모드',
                                style: TextStyle(
                                  color: clrSuccess,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                            ],
                          ),
                          const SizedBox(height: 30),
                          Row(
                            children: [
                              Expanded(
                                child: Column(
                                  crossAxisAlignment: CrossAxisAlignment.start,
                                  children: [
                                    Text(
                                      isStopping
                                          ? '중지 신호 처리 중'
                                          : t('foldingNow'),
                                      style: const TextStyle(
                                        fontSize: 25,
                                        color: clrTextMain,
                                        fontWeight: FontWeight.w900,
                                      ),
                                    ),
                                    const SizedBox(height: 12),
                                    const Text(
                                      'Jetson 실제 완료 수량',
                                      style: TextStyle(
                                        fontSize: 13,
                                        color: clrTextSub,
                                      ),
                                    ),
                                    Text(
                                      '$currentCount / $_activeTarget 벌',
                                      style: TextStyle(
                                        fontSize: 28,
                                        color: clrMainBlue,
                                        fontWeight: FontWeight.w800,
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                              CuteRobot(
                                state: isStopping
                                    ? RobotState.paused
                                    : RobotState.working,
                              ),
                            ],
                          ),
                          const SizedBox(height: 20),
                          Text(
                            isStopping
                                ? '현장의 로봇이 안전하게 정지했다는 확인 신호를 기다리고 있습니다.'
                                : '로봇이 완료 신호를 보낼 때마다 수량이 바뀝니다.\n'
                                      '모든 작업을 마치면 Jetson이 종료를 알려줍니다.',
                            style: const TextStyle(
                              color: clrTextSub,
                              fontSize: 13,
                              height: 1.5,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                  const SizedBox(height: 14),
                  OutlinedButton.icon(
                    onPressed: isStopping || !_canControl || _commandPending
                        ? null
                        : requestStop,
                    icon: const Icon(Icons.stop_circle_outlined),
                    label: Padding(
                      padding: const EdgeInsets.symmetric(vertical: 13),
                      child: Text(isStopping ? '정지 확인 대기 중' : '작업 중지 요청'),
                    ),
                    style: OutlinedButton.styleFrom(foregroundColor: clrDanger),
                  ),
                  const SizedBox(height: 12),
                  const Text(
                    '앱의 중지 요청은 물리 비상정지 버튼이 아닙니다.',
                    textAlign: TextAlign.center,
                    style: TextStyle(fontSize: 11, color: clrTextSub),
                  ),
                ],
                const SizedBox(height: 26),
                TextButton.icon(
                  onPressed: showDataDashboard,
                  icon: const Icon(Icons.insights, color: clrTextSub),
                  label: Text(
                    t('statsBtn'),
                    style: const TextStyle(color: clrTextSub),
                  ),
                ),
              ],
            ),
          ),
        ),
        if (_showTutorial) _buildTutorialOverlay(),
      ],
    );
  }
}

// ======================================================================
// 👔 저비의 코디룸 (가짜 테스트 기능 배제된 원본 + 최신 구글 제미나이 연동)
// ======================================================================
class JeobyCodyScreen extends StatefulWidget {
  final String langCode;
  const JeobyCodyScreen({super.key, required this.langCode});

  @override
  State<JeobyCodyScreen> createState() => _JeobyCodyScreenState();
}

class _JeobyCodyScreenState extends State<JeobyCodyScreen> {
  int _introStep = 0;
  String _currentWeather = 'sunny';
  String _currentTemp = '--°';
  double _tempValue = 20.0;
  bool _isLoadingWeather = true;
  String _currentLocationName = '위치 확인 중';

  String _aiMessage = '날씨를 확인하고 옷장을 스캔 중입니다... 🤖';
  bool _isThinking = false;

  final List<Map<String, dynamic>> _colorPalette = [
    {'name': '흰색', 'color': Colors.white},
    {'name': '검정', 'color': const Color(0xFF1E293B)},
    {'name': '회색', 'color': Colors.grey},
    {'name': '네이비', 'color': const Color(0xFF1E3A8A)},
    {'name': '베이지', 'color': const Color(0xFFD4D4D8)},
    {'name': '청(데님)', 'color': const Color(0xFF3B82F6)},
  ];

  List<Map<String, dynamic>> myClothes = [
    {
      'name': '기본 면 반팔',
      'type': '상의',
      'emoji': '👕',
      'colorInfo': {'name': '흰색', 'color': Colors.white},
    },
    {
      'name': '시원한 린넨',
      'type': '하의',
      'emoji': '👖',
      'colorInfo': {'name': '베이지', 'color': const Color(0xFFD4D4D8)},
    },
    {
      'name': '와이드 슬랙스',
      'type': '하의',
      'emoji': '👖',
      'colorInfo': {'name': '검정', 'color': const Color(0xFF1E293B)},
    },
  ];

  String t(String key) {
    try {
      return getTrans(widget.langCode, key);
    } catch (e) {
      return key;
    }
  }

  @override
  void initState() {
    super.initState();
    _fetchWeather();
  }

  Future<Position?> _getCurrentPosition() async {
    try {
      final serviceEnabled = await Geolocator.isLocationServiceEnabled();
      if (!serviceEnabled) {
        debugPrint('위치 서비스가 꺼져 있어 서울을 기본 위치로 사용합니다.');
        return null;
      }

      var permission = await Geolocator.checkPermission();
      if (permission == LocationPermission.denied) {
        permission = await Geolocator.requestPermission();
      }

      if (permission == LocationPermission.denied ||
          permission == LocationPermission.deniedForever) {
        debugPrint('위치 권한이 없어 서울을 기본 위치로 사용합니다.');
        return null;
      }

      return await Geolocator.getCurrentPosition(
        locationSettings: const LocationSettings(
          accuracy: LocationAccuracy.high,
          timeLimit: Duration(seconds: 10),
        ),
      );
    } catch (e) {
      debugPrint('현재 위치 가져오기 실패: $e');
      return null;
    }
  }

  Future<String> _getLocationName(double latitude, double longitude) async {
    try {
      final url = Uri.parse(
        'https://api.bigdatacloud.net/data/reverse-geocode-client'
        '?latitude=$latitude&longitude=$longitude&localityLanguage=ko',
      );

      final response = await http.get(url).timeout(const Duration(seconds: 8));
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        final candidates = [
          data['locality'],
          data['city'],
          data['principalSubdivision'],
        ];

        for (final candidate in candidates) {
          if (candidate != null && candidate.toString().trim().isNotEmpty) {
            return candidate.toString().trim();
          }
        }
      }
    } catch (e) {
      debugPrint('지역명 가져오기 실패: $e');
    }

    return '현재 위치';
  }

  Future<void> _fetchWeather() async {
    if (mounted) {
      setState(() {
        _isLoadingWeather = true;
        _currentLocationName = '위치 확인 중';
      });
    }

    // 위치 권한 거부·위치 서비스 꺼짐·시간 초과 시 사용할 기본값입니다.
    double latitude = 37.566;
    double longitude = 126.9784;
    String locationName = '서울 (기본 위치)';

    try {
      final position = await _getCurrentPosition();
      if (position != null) {
        latitude = position.latitude;
        longitude = position.longitude;
        final resolvedName = await _getLocationName(latitude, longitude);
        locationName = '$resolvedName (현재)';
      }

      final url = Uri.parse(
        'https://api.open-meteo.com/v1/forecast'
        '?latitude=$latitude&longitude=$longitude'
        '&current_weather=true&timezone=auto',
      );

      final response = await http.get(url).timeout(const Duration(seconds: 8));
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        final temp = data['current_weather']['temperature'];
        final weatherCode = data['current_weather']['weathercode'];

        if (!mounted) return;
        setState(() {
          _currentLocationName = locationName;
          _tempValue = temp.toDouble();
          _currentTemp = '${temp.round()}°';
          _isLoadingWeather = false;

          if (weatherCode == 1 ||
              weatherCode == 2 ||
              weatherCode == 3 ||
              weatherCode == 45 ||
              weatherCode == 48)
            _currentWeather = 'cloudy';
          else if ((weatherCode >= 51 && weatherCode <= 67) ||
              (weatherCode >= 80 && weatherCode <= 82) ||
              (weatherCode >= 95 && weatherCode <= 99))
            _currentWeather = 'rainy';
          else if ((weatherCode >= 71 && weatherCode <= 77) ||
              (weatherCode >= 85 && weatherCode <= 86))
            _currentWeather = 'snowy';
          else
            _currentWeather = 'sunny';
        });

        if (mounted) _askJeobyAI();
      } else {
        if (!mounted) return;
        setState(() {
          _currentLocationName = locationName;
          _currentTemp = '확인 실패';
          _isLoadingWeather = false;
        });
      }
    } catch (e) {
      debugPrint('현재 위치 날씨 가져오기 실패: $e');
      if (!mounted) return;
      setState(() {
        _currentLocationName = locationName;
        _currentTemp = '확인 실패';
        _isLoadingWeather = false;
      });
    }
  }

  Future<void> _askJeobyAI() async {
    if (!mounted || _isThinking) return;
    if (myClothes.isEmpty) {
      setState(() => _aiMessage = '옷장에 옷이 없어요!\n[옷 등록]을 눌러 채워주세요.');
      return;
    }

    setState(() {
      _isThinking = true;
      _aiMessage = '저비가 인공지능 두뇌를 풀가동 중입니다...\n(잠시만 기다려주세요 🧠✨)';
    });

    try {
      const apiKey = String.fromEnvironment('GEMINI_API_KEY');

      if (apiKey.isEmpty) {
        setState(() {
          _aiMessage =
              'Gemini API Key가 설정되지 않았습니다.\n'
              '실행할 때 --dart-define=GEMINI_API_KEY=... 를 추가해주세요.';
          _isThinking = false;
        });
        return;
      }

      final model = GenerativeModel(
        model: 'gemini-3-flash-preview',
        apiKey: apiKey,
      );

      String clothesList = myClothes
          .map(
            (c) =>
                '[${c['type']}] ${c['name']} (색상: ${c['colorInfo']['name']})',
          )
          .join(', ');

      final prompt =
          '''
      너는 똑똑하고 센스 있는 패션 코디 로봇 '저비(Jeoby)'야. 친근하게 반말을 사용해.
      현재 날씨: $_currentWeather (sunny:맑음, cloudy:흐림, rainy:비, snowy:눈)
      현재 기온: $_tempValue도.
      내 옷장 리스트: $clothesList

      조건:
      1. 반드시 '내 옷장 리스트'에 있는 옷 중에서만 상의 1개와 하의 1개를 골라서 조합해줘.
      2. 온도와 날씨 상태를 고려해서 가장 현실적인 옷을 골라.
      3. 답변은 3~4줄 이내로, 선택한 상하의 이름과 왜 이렇게 추천했는지 유쾌하게 설명해줘.
      4. 강조할때 별 "**" 이런 내용 넣지말고 설명해줘.
      ''';

      final content = [Content.text(prompt)];
      final response = await model
          .generateContent(content)
          .timeout(const Duration(seconds: 25));
      if (!mounted) return;

      setState(() {
        _aiMessage = response.text?.trim() ?? '저비의 회로가 잠시 꼬였어요! 다시 시도해주세요.';
        _isThinking = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _aiMessage = '에러 원인: ${e.toString()}';
        _isThinking = false;
      });
    }
  }

  void _showAddClothDialog() {
    String newName = '';
    String newType = '상의';
    Map<String, dynamic> selectedColor = _colorPalette[0];

    showDialog(
      context: context,
      builder: (context) => StatefulBuilder(
        builder: (context, setDialogState) {
          return AlertDialog(
            backgroundColor: Colors.white,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(15),
            ),
            title: const Text(
              '내 옷장 채우기 👗',
              style: TextStyle(
                fontWeight: FontWeight.bold,
                color: Color(0xFF1E293B),
              ),
            ),
            content: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                TextField(
                  decoration: const InputDecoration(
                    labelText: '어떤 옷인가요? (예: 두꺼운 니트)',
                  ),
                  onChanged: (val) => newName = val,
                ),
                const SizedBox(height: 15),
                DropdownButtonFormField<String>(
                  value: newType,
                  decoration: const InputDecoration(labelText: '옷 종류'),
                  items: const [
                    DropdownMenuItem(value: '상의', child: Text('상의 👕')),
                    DropdownMenuItem(value: '하의', child: Text('하의 👖')),
                  ],
                  onChanged: (val) => setDialogState(() => newType = val!),
                ),
                const SizedBox(height: 15),
                const Text(
                  '대표 색상',
                  style: TextStyle(fontSize: 12, color: Colors.grey),
                ),
                const SizedBox(height: 5),
                Wrap(
                  spacing: 10,
                  children: _colorPalette.map((colorMap) {
                    bool isSelected = selectedColor['name'] == colorMap['name'];
                    return GestureDetector(
                      onTap: () =>
                          setDialogState(() => selectedColor = colorMap),
                      child: Container(
                        width: 30,
                        height: 30,
                        decoration: BoxDecoration(
                          color: colorMap['color'],
                          shape: BoxShape.circle,
                          border: Border.all(
                            color: isSelected
                                ? Colors.blueAccent
                                : Colors.grey.shade300,
                            width: isSelected ? 3 : 1,
                          ),
                          boxShadow: isSelected
                              ? [
                                  const BoxShadow(
                                    color: Colors.blueAccent,
                                    blurRadius: 4,
                                  ),
                                ]
                              : [],
                        ),
                      ),
                    );
                  }).toList(),
                ),
              ],
            ),
            actions: [
              TextButton(
                onPressed: () => Navigator.pop(context),
                child: const Text('취소', style: TextStyle(color: Colors.grey)),
              ),
              ElevatedButton(
                onPressed: () {
                  if (newName.isNotEmpty) {
                    String emoji = newType == '상의' ? '👕' : '👖';
                    setState(() {
                      myClothes.add({
                        'name': newName,
                        'type': newType,
                        'emoji': emoji,
                        'colorInfo': selectedColor,
                      });
                    });
                    Navigator.pop(context);

                    if (_introStep == 2) _askJeobyAI();
                  }
                },
                style: ElevatedButton.styleFrom(
                  backgroundColor: const Color(0xFF2563EB),
                  foregroundColor: Colors.white,
                ),
                child: const Text('등록'),
              ),
            ],
          );
        },
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFE2E8F0),
      appBar: _introStep == 2
          ? AppBar(
              backgroundColor: Colors.transparent,
              elevation: 0,
              iconTheme: const IconThemeData(color: Color(0xFF1E293B)),
              title: Text(
                t('menuCody'),
                style: const TextStyle(
                  fontWeight: FontWeight.bold,
                  color: Color(0xFF1E293B),
                ),
              ),
              actions: [
                Padding(
                  padding: const EdgeInsets.only(right: 20.0),
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    crossAxisAlignment: CrossAxisAlignment.end,
                    children: [
                      Text(
                        '📍 $_currentLocationName',
                        style: const TextStyle(
                          fontSize: 11,
                          color: Color(0xFF64748B),
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      _isLoadingWeather
                          ? const SizedBox(
                              width: 15,
                              height: 15,
                              child: CircularProgressIndicator(strokeWidth: 2),
                            )
                          : Text(
                              '기온 $_currentTemp',
                              style: const TextStyle(
                                fontSize: 14,
                                color: Color(0xFF2563EB),
                                fontWeight: FontWeight.w900,
                              ),
                            ),
                    ],
                  ),
                ),
              ],
            )
          : null,
      body: Stack(
        children: [
          if (_introStep == 2)
            Column(
              children: [
                const SizedBox(height: 10),
                Expanded(
                  child: Stack(
                    alignment: Alignment.center,
                    children: [
                      Positioned(
                        top: 10,
                        child: Container(
                          width: 240,
                          height: 150,
                          decoration: BoxDecoration(
                            color: _currentWeather == 'sunny'
                                ? const Color(0xFFBAE6FD)
                                : _currentWeather == 'cloudy'
                                ? const Color(0xFFCBD5E1)
                                : _currentWeather == 'rainy'
                                ? const Color(0xFF94A3B8)
                                : const Color(0xFFF1F5F9),
                            border: Border.all(color: Colors.white, width: 8),
                            borderRadius: BorderRadius.circular(10),
                            boxShadow: const [
                              BoxShadow(
                                color: Colors.black26,
                                blurRadius: 10,
                                offset: Offset(0, 5),
                              ),
                            ],
                          ),
                          child: Stack(
                            alignment: Alignment.center,
                            children: [
                              if (_currentWeather == 'sunny')
                                const Icon(
                                  Icons.wb_sunny,
                                  color: Colors.orangeAccent,
                                  size: 70,
                                ),
                              if (_currentWeather == 'cloudy')
                                const Icon(
                                  Icons.cloud,
                                  color: Colors.white70,
                                  size: 70,
                                ),
                              Container(
                                width: double.infinity,
                                height: 6,
                                color: Colors.white,
                              ),
                              Container(
                                width: 6,
                                height: double.infinity,
                                color: Colors.white,
                              ),
                            ],
                          ),
                        ),
                      ),
                      Positioned.fill(
                        child: WeatherParticles(weather: _currentWeather),
                      ),
                      Positioned(
                        top: 180,
                        left: 20,
                        right: 20,
                        child: Container(
                          height: 160,
                          padding: const EdgeInsets.all(15),
                          decoration: BoxDecoration(
                            color: Colors.white.withOpacity(0.95),
                            borderRadius: BorderRadius.circular(20),
                            boxShadow: const [
                              BoxShadow(
                                color: Colors.black12,
                                blurRadius: 10,
                                offset: Offset(0, 5),
                              ),
                            ],
                          ),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Row(
                                mainAxisAlignment:
                                    MainAxisAlignment.spaceBetween,
                                children: [
                                  const Text(
                                    '나의 옷장 👗',
                                    style: TextStyle(
                                      fontSize: 16,
                                      fontWeight: FontWeight.bold,
                                      color: Color(0xFF1E293B),
                                    ),
                                  ),
                                  ElevatedButton.icon(
                                    onPressed: _showAddClothDialog,
                                    style: ElevatedButton.styleFrom(
                                      backgroundColor: const Color(0xFF2563EB),
                                      foregroundColor: Colors.white,
                                      padding: const EdgeInsets.symmetric(
                                        horizontal: 12,
                                        vertical: 6,
                                      ),
                                      minimumSize: Size.zero,
                                    ),
                                    icon: const Icon(Icons.add, size: 16),
                                    label: const Text(
                                      '옷 등록',
                                      style: TextStyle(fontSize: 13),
                                    ),
                                  ),
                                ],
                              ),
                              const SizedBox(height: 10),
                              Expanded(
                                child: myClothes.isEmpty
                                    ? const Center(
                                        child: Text(
                                          '등록된 옷이 없어요!\n[옷 등록]을 눌러 추가해보세요.',
                                          textAlign: TextAlign.center,
                                          style: TextStyle(
                                            color: Colors.grey,
                                            fontSize: 13,
                                          ),
                                        ),
                                      )
                                    : ListView.builder(
                                        scrollDirection: Axis.horizontal,
                                        itemCount: myClothes.length,
                                        itemBuilder: (context, index) {
                                          final cloth = myClothes[index];
                                          final colorInfo = cloth['colorInfo'];
                                          return Stack(
                                            clipBehavior: Clip.none,
                                            children: [
                                              Container(
                                                width: 90,
                                                margin: const EdgeInsets.only(
                                                  right: 15,
                                                  top: 5,
                                                ),
                                                decoration: BoxDecoration(
                                                  color: const Color(
                                                    0xFFF8FAFC,
                                                  ),
                                                  borderRadius:
                                                      BorderRadius.circular(12),
                                                  border: Border.all(
                                                    color: const Color(
                                                      0xFFE2E8F0,
                                                    ),
                                                  ),
                                                ),
                                                child: Column(
                                                  mainAxisAlignment:
                                                      MainAxisAlignment.center,
                                                  children: [
                                                    Text(
                                                      cloth['emoji'],
                                                      style: const TextStyle(
                                                        fontSize: 26,
                                                      ),
                                                    ),
                                                    const SizedBox(height: 5),
                                                    Row(
                                                      mainAxisAlignment:
                                                          MainAxisAlignment
                                                              .center,
                                                      children: [
                                                        Container(
                                                          width: 10,
                                                          height: 10,
                                                          decoration: BoxDecoration(
                                                            color:
                                                                colorInfo['color'],
                                                            shape:
                                                                BoxShape.circle,
                                                            border: Border.all(
                                                              color: Colors
                                                                  .grey
                                                                  .shade400,
                                                              width: 0.5,
                                                            ),
                                                          ),
                                                        ),
                                                        const SizedBox(
                                                          width: 4,
                                                        ),
                                                        Text(
                                                          cloth['type'],
                                                          style:
                                                              const TextStyle(
                                                                fontSize: 11,
                                                                color: Color(
                                                                  0xFF64748B,
                                                                ),
                                                              ),
                                                        ),
                                                      ],
                                                    ),
                                                    Padding(
                                                      padding:
                                                          const EdgeInsets.symmetric(
                                                            horizontal: 4.0,
                                                            vertical: 2.0,
                                                          ),
                                                      child: Text(
                                                        cloth['name'],
                                                        textAlign:
                                                            TextAlign.center,
                                                        maxLines: 2,
                                                        overflow: TextOverflow
                                                            .ellipsis,
                                                        style: const TextStyle(
                                                          fontSize: 12,
                                                          fontWeight:
                                                              FontWeight.bold,
                                                          color: Color(
                                                            0xFF1E293B,
                                                          ),
                                                        ),
                                                      ),
                                                    ),
                                                  ],
                                                ),
                                              ),
                                              Positioned(
                                                top: -2,
                                                right: 8,
                                                child: GestureDetector(
                                                  onTap: () {
                                                    setState(
                                                      () => myClothes.removeAt(
                                                        index,
                                                      ),
                                                    );
                                                    if (_introStep == 2)
                                                      _askJeobyAI();
                                                  },
                                                  child: Container(
                                                    decoration:
                                                        const BoxDecoration(
                                                          color: Colors.white,
                                                          shape:
                                                              BoxShape.circle,
                                                        ),
                                                    child: const Icon(
                                                      Icons.cancel,
                                                      size: 20,
                                                      color: Colors.redAccent,
                                                    ),
                                                  ),
                                                ),
                                              ),
                                            ],
                                          );
                                        },
                                      ),
                              ),
                            ],
                          ),
                        ),
                      ),

                      // 💬 구글 AI가 보내준 답변이 여기에 뜹니다!
                      Positioned(
                        bottom: 160,
                        child: GestureDetector(
                          onTap: _isThinking ? null : _askJeobyAI,
                          child: Container(
                            padding: const EdgeInsets.all(20),
                            width: MediaQuery.of(context).size.width * 0.85,
                            constraints: const BoxConstraints(minHeight: 80),
                            decoration: BoxDecoration(
                              color: Colors.white,
                              borderRadius: BorderRadius.circular(20),
                              boxShadow: const [
                                BoxShadow(
                                  color: Colors.black12,
                                  blurRadius: 10,
                                ),
                              ],
                            ),
                            child: _isThinking
                                ? const Column(
                                    mainAxisSize: MainAxisSize.min,
                                    children: [
                                      SizedBox(
                                        width: 20,
                                        height: 20,
                                        child: CircularProgressIndicator(
                                          strokeWidth: 2,
                                        ),
                                      ),
                                      SizedBox(height: 10),
                                      Text(
                                        '저비가 날씨 맞춤 코디를\n준비 중입니다... 🧠✨',
                                        textAlign: TextAlign.center,
                                        style: TextStyle(
                                          color: Colors.grey,
                                          fontWeight: FontWeight.bold,
                                        ),
                                      ),
                                    ],
                                  )
                                : Text(
                                    _aiMessage,
                                    textAlign: TextAlign.center,
                                    style: const TextStyle(
                                      fontSize: 14,
                                      height: 1.5,
                                      color: Color(0xFF1E293B),
                                      fontWeight: FontWeight.bold,
                                    ),
                                  ),
                          ),
                        ),
                      ),
                      Positioned(
                        bottom: 20,
                        child: CuteRobot(
                          state: _isThinking
                              ? RobotState.working
                              : RobotState.idle,
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),

          if (_introStep < 2)
            Positioned.fill(
              child: Material(
                color: Colors.black.withOpacity(0.85),
                child: SafeArea(
                  child: Center(
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Container(
                          margin: const EdgeInsets.symmetric(horizontal: 40),
                          padding: const EdgeInsets.all(25),
                          decoration: BoxDecoration(
                            color: Colors.white,
                            borderRadius: BorderRadius.circular(20),
                          ),
                          child: Text(
                            _introStep == 0 ? t('codyIntro1') : t('codyIntro2'),
                            textAlign: TextAlign.center,
                            style: const TextStyle(
                              fontSize: 18,
                              fontWeight: FontWeight.bold,
                              color: Color(0xFF1E293B),
                              height: 1.5,
                            ),
                          ),
                        ),
                        const Icon(
                          Icons.arrow_drop_down,
                          color: Colors.white,
                          size: 60,
                        ),
                        const CuteRobot(state: RobotState.idle),
                        const SizedBox(height: 60),
                        ElevatedButton(
                          onPressed: () => setState(() => _introStep++),
                          style: ElevatedButton.styleFrom(
                            backgroundColor: const Color(0xFF2563EB),
                            foregroundColor: Colors.white,
                            padding: const EdgeInsets.symmetric(
                              horizontal: 40,
                              vertical: 15,
                            ),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(30),
                            ),
                          ),
                          child: Text(
                            _introStep == 0 ? t('t1Next') : t('codyStart'),
                            style: const TextStyle(
                              fontSize: 18,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ),
        ],
      ),
    );
  }
}

// ======================================================================
// 🌧️ ❄️ 날씨 애니메이션 위젯 (수정 없음)
// ======================================================================
class WeatherParticles extends StatefulWidget {
  final String weather;
  const WeatherParticles({super.key, required this.weather});
  @override
  State<WeatherParticles> createState() => _WeatherParticlesState();
}

class _WeatherParticlesState extends State<WeatherParticles>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1500),
    )..repeat();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (widget.weather == 'sunny' || widget.weather == 'cloudy')
      return const SizedBox();
    bool isRain = widget.weather == 'rainy';
    return AnimatedBuilder(
      animation: _controller,
      builder: (context, child) {
        return Stack(
          children: List.generate(20, (index) {
            double startX = (index * 37.0) % MediaQuery.of(context).size.width;
            double progress = (_controller.value + (index * 0.1)) % 1.0;
            double startY =
                -20.0 + (progress * MediaQuery.of(context).size.height);
            return Positioned(
              left: startX,
              top: startY,
              child: Icon(
                isRain ? Icons.water_drop : Icons.ac_unit,
                color: isRain
                    ? Colors.blueAccent.withOpacity(0.6)
                    : Colors.white.withOpacity(0.9),
                size: isRain ? 12 : 16,
              ),
            );
          }),
        );
      },
    );
  }
}

// ======================================================================
// 🤖 저비(Jeoby) 캐릭터 애니메이션 (수정 없음)
// ======================================================================
enum RobotState { idle, working, paused }

class CuteRobot extends StatefulWidget {
  final RobotState state;
  const CuteRobot({super.key, required this.state});
  @override
  State<CuteRobot> createState() => _CuteRobotState();
}

class _CuteRobotState extends State<CuteRobot>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: Duration(
        milliseconds: widget.state == RobotState.working ? 350 : 1500,
      ),
    );
    if (widget.state != RobotState.paused) _controller.repeat();
  }

  @override
  void didUpdateWidget(CuteRobot oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.state != oldWidget.state) {
      if (widget.state == RobotState.working) {
        _controller.duration = const Duration(milliseconds: 350);
        _controller.repeat();
      } else if (widget.state == RobotState.paused) {
        _controller.stop();
      } else {
        _controller.duration = const Duration(milliseconds: 1500);
        _controller.repeat();
      }
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _controller,
      builder: (context, child) {
        final isWorking = widget.state == RobotState.working;
        final isPaused = widget.state == RobotState.paused;
        final bounce = isPaused
            ? 0.0
            : math.sin(_controller.value * math.pi * 2) *
                  (isWorking ? 8.0 : 3.0);
        final sweatDropY = _controller.value * 35.0;
        final sweatOpacity = isWorking
            ? (1.0 - _controller.value).clamp(0.0, 1.0)
            : 0.0;
        final armRotation = isWorking
            ? math.sin(_controller.value * math.pi * 2) * 0.8
            : 0.0;
        return SizedBox(
          width: 140,
          height: 120,
          child: Transform.translate(
            offset: Offset(0, bounce),
            child: Stack(
              clipBehavior: Clip.none,
              alignment: Alignment.center,
              children: [
                Positioned(
                  top: 0,
                  child: Column(
                    children: [
                      Container(
                        width: 12,
                        height: 12,
                        decoration: BoxDecoration(
                          color: isWorking
                              ? Colors.redAccent
                              : const Color(0xFF38BDF8),
                          shape: BoxShape.circle,
                          boxShadow: [
                            BoxShadow(
                              color:
                                  (isWorking
                                          ? Colors.redAccent
                                          : const Color(0xFF38BDF8))
                                      .withOpacity(0.5),
                              blurRadius: 8,
                              spreadRadius: 2,
                            ),
                          ],
                        ),
                      ),
                      Container(
                        width: 4,
                        height: 15,
                        color: const Color(0xFF94A3B8),
                      ),
                    ],
                  ),
                ),
                Positioned(
                  left: 5,
                  top: 55,
                  child: Transform.rotate(
                    angle: armRotation,
                    alignment: Alignment.centerRight,
                    child: Container(
                      width: 40,
                      height: 16,
                      decoration: BoxDecoration(
                        color: const Color(0xFF94A3B8),
                        borderRadius: BorderRadius.circular(10),
                      ),
                    ),
                  ),
                ),
                Positioned(
                  right: 5,
                  top: 55,
                  child: Transform.rotate(
                    angle: -armRotation,
                    alignment: Alignment.centerLeft,
                    child: Container(
                      width: 40,
                      height: 16,
                      decoration: BoxDecoration(
                        color: const Color(0xFF94A3B8),
                        borderRadius: BorderRadius.circular(10),
                      ),
                    ),
                  ),
                ),
                Positioned(
                  top: 25,
                  child: Container(
                    width: 90,
                    height: 75,
                    decoration: BoxDecoration(
                      color: const Color(0xFFE2E8F0),
                      borderRadius: BorderRadius.circular(20),
                      boxShadow: const [
                        BoxShadow(
                          color: Colors.black12,
                          blurRadius: 10,
                          offset: Offset(0, 5),
                        ),
                      ],
                      border: Border.all(
                        color: const Color(0xFFCBD5E1),
                        width: 2,
                      ),
                    ),
                    child: Center(
                      child: Container(
                        width: 70,
                        height: 40,
                        decoration: BoxDecoration(
                          color: const Color(0xFF0F172A),
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: Row(
                          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                          children: [
                            _buildEye(isPaused, isWorking),
                            _buildEye(isPaused, isWorking),
                          ],
                        ),
                      ),
                    ),
                  ),
                ),
                if (isWorking)
                  Positioned(
                    top: 20 + sweatDropY,
                    right: 15,
                    child: Opacity(
                      opacity: sweatOpacity,
                      child: const Icon(
                        Icons.water_drop,
                        color: Colors.lightBlue,
                        size: 26,
                      ),
                    ),
                  ),
              ],
            ),
          ),
        );
      },
    );
  }

  Widget _buildEye(bool isPaused, bool isWorking) {
    if (isPaused)
      return Container(
        width: 18,
        height: 4,
        decoration: BoxDecoration(
          color: const Color(0xFF38BDF8),
          borderRadius: BorderRadius.circular(2),
        ),
      );
    return AnimatedContainer(
      duration: const Duration(milliseconds: 200),
      width: isWorking ? 18 : 14,
      height: isWorking ? 18 : 14,
      decoration: const BoxDecoration(
        color: Color(0xFF38BDF8),
        shape: BoxShape.circle,
      ),
    );
  }
}
