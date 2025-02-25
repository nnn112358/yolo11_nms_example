import onnxruntime as ort
import cv2
import numpy as np
import argparse
import os

# COCOデータセットのキーポイント定義（17点）
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", 
    "left_wrist", "right_wrist", "left_hip", "right_hip", 
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# キーポイント間の接続定義（骨格線）
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2),  # 鼻 - 目
    (1, 3), (2, 4),  # 目 - 耳
    (5, 7), (7, 9),  # 左肩 - 左肘 - 左手首
    (6, 8), (8, 10),  # 右肩 - 右肘 - 右手首
    (5, 6), (5, 11), (6, 12),  # 肩 - 腰
    (11, 13), (13, 15),  # 左腰 - 左膝 - 左足首
    (12, 14), (14, 16)  # 右腰 - 右膝 - 右足首
]

# パーツごとに異なる色を定義
KEYPOINT_COLORS = [
    (255, 0, 0),    # 0: nose (赤)
    (255, 85, 0),   # 1: left_eye (オレンジ)
    (255, 170, 0),  # 2: right_eye (薄いオレンジ)
    (255, 255, 0),  # 3: left_ear (黄)
    (170, 255, 0),  # 4: right_ear (黄緑)
    (85, 255, 0),   # 5: left_shoulder (緑)
    (0, 255, 0),    # 6: right_shoulder (緑)
    (0, 255, 85),   # 7: left_elbow (青緑)
    (0, 255, 170),  # 8: right_elbow (青緑)
    (0, 255, 255),  # 9: left_wrist (水色)
    (0, 170, 255),  # 10: right_wrist (水色)
    (0, 85, 255),   # 11: left_hip (青)
    (0, 0, 255),    # 12: right_hip (青)
    (85, 0, 255),   # 13: left_knee (紫)
    (170, 0, 255),  # 14: right_knee (紫)
    (255, 0, 255),  # 15: left_ankle (マゼンタ)
    (255, 0, 170)   # 16: right_ankle (ピンク)
]

# 骨格線の色を定義
LIMB_COLORS = [
    (255, 128, 0),  # 鼻 - 目 (オレンジ)
    (255, 128, 0),
    (255, 200, 0),  # 目 - 耳 (黄色)
    (255, 200, 0),
    (0, 255, 0),    # 左肩 - 左肘 - 左手首 (緑)
    (0, 255, 150),
    (0, 255, 0),    # 右肩 - 右肘 - 右手首 (緑)
    (0, 255, 150),
    (0, 150, 255),  # 肩 - 腰 (水色)
    (0, 100, 255),
    (0, 100, 255),
    (130, 0, 255),  # 左腰 - 左膝 - 左足首 (紫)
    (230, 0, 255),
    (130, 0, 255),  # 右腰 - 右膝 - 右足首 (紫)
    (230, 0, 255)
]

def load_model(model_path):
    """ONNXモデルをロードする関数"""
    try:
        # プロバイダーを明示的に指定
        providers = ['CPUExecutionProvider']
        session = ort.InferenceSession(model_path, providers=providers)
        print(f"モデルを正常にロードしました: {model_path}")
        return session
    except Exception as e:
        print(f"モデルのロード中にエラーが発生しました: {e}")
        return None

def prepare_image(image_path, input_size=640):
    """画像を前処理する関数"""
    # 画像を読み込む
    img = cv2.imread(image_path)
    if img is None:
        print(f"画像の読み込みに失敗しました: {image_path}")
        return None, None
    
    # 元の画像サイズを保存
    original_height, original_width = img.shape[:2]
    
    # 画像をRGBに変換
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 画像をリサイズ
    img_resized = cv2.resize(img_rgb, (input_size, input_size))
    
    # 画像を前処理（正規化、次元の追加、転置）
    input_data = img_resized.transpose(2, 0, 1).astype(np.float32) / 255.0
    input_data = np.expand_dims(input_data, axis=0)
    
    return input_data, (original_width, original_height, img)

def run_inference(session, input_data):
    """モデルで推論を実行する関数"""
    try:
        # 入力名を取得
        input_name = session.get_inputs()[0].name
        
        # 推論を実行
        outputs = session.run(None, {input_name: input_data})
        return outputs
    except Exception as e:
        print(f"推論中にエラーが発生しました: {e}")
        return None

def process_pose_detections(outputs, original_size, input_size=640, conf_threshold=0.25, kpt_conf_threshold=0.2):
    """
    ポーズ検出結果を処理して表示する関数
    """
    original_width, original_height, original_img = original_size
    
    # 出力が複数ある場合があるため、適切な出力を使用
    # YOLOv8-poseのNMS後の出力は[1, 300, 57]形式であることを想定
    # 各検出には17個のキーポイント（x, y, conf）× 17 = 51が含まれる
    detections = outputs[0]  # 最初の出力を使用
    print(f"検出形状: {detections.shape}")
    
    # 閾値を超える検出結果のみを保持
    valid_detections = []
    for detection in detections[0]:  # バッチ次元を削除
        # ボックス信頼度 (4番目の値) がconf_thresholdを超える場合のみ処理
        score = detection[4]
        if score > conf_threshold:
            valid_detections.append(detection)
    
    result_img = original_img.copy()
    
    # 入力サイズから元の画像サイズへのスケーリング係数
    scale_x = original_width / input_size
    scale_y = original_height / input_size
    
    # 検出結果を描画
    for i, detection in enumerate(valid_detections):
        # ボックス座標（xyxy形式）とスコア、クラス
        x1, y1, x2, y2 = detection[:4]
        score = detection[4]
        class_id = int(detection[5])
        
        # 入力サイズから元の画像サイズにスケーリング
        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)
        
        # 座標が画像範囲内に収まるように調整
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(original_width - 1, x2)
        y2 = min(original_height - 1, y2)
        
        # 境界ボックスを描画（半透明の背景）
        overlay = result_img.copy()
        box_color = (0, 255, 0)  # 緑色
        cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, 2)
        
        # 半透明の効果を適用
        alpha = 0.3  # 透明度
        cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, -1)
        cv2.addWeighted(overlay, alpha, result_img, 1 - alpha, 0, result_img)
        
        # スコアとクラスIDを表示
        label = f"Person: {score:.2f}"
        cv2.putText(
            result_img,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            box_color,
            2,
            cv2.LINE_AA
        )
        
        # キーポイントの処理（インデックス6から始まる）
        keypoints = []
        for kpt_idx in range(17):
            # キーポイントのインデックス計算（各キーポイントはx, y, confの3つの値）
            base_idx = 6 + (kpt_idx * 3)
            kx, ky, kp_conf = detection[base_idx:base_idx+3]
            
            # キーポイント座標をスケーリング
            kx_scaled = int(kx * scale_x)
            ky_scaled = int(ky * scale_y)
            
            # 信頼度が閾値以上の場合のみ保存
            keypoints.append((kx_scaled, ky_scaled, kp_conf))
        
        # 骨格線を描画（キーポイント間の接続）
        for limb_idx, connection in enumerate(SKELETON_CONNECTIONS):
            start_idx, end_idx = connection
            start_keypoint = keypoints[start_idx]
            end_keypoint = keypoints[end_idx]
            
            # 両端のキーポイントが信頼度閾値を超える場合のみ描画
            if start_keypoint[2] > kpt_conf_threshold and end_keypoint[2] > kpt_conf_threshold:
                color = LIMB_COLORS[limb_idx]
                start_pt = (start_keypoint[0], start_keypoint[1])
                end_pt = (end_keypoint[0], end_keypoint[1])
                
                cv2.line(result_img, start_pt, end_pt, color, 2, cv2.LINE_AA)
        
        # キーポイントを描画
        for kpt_idx, (kx_scaled, ky_scaled, kp_conf) in enumerate(keypoints):
            # 信頼度が閾値以上の場合のみ描画
            if kp_conf > kpt_conf_threshold:
                color = KEYPOINT_COLORS[kpt_idx]
                
                # キーポイントを円で描画
                cv2.circle(result_img, (kx_scaled, ky_scaled), 4, color, -1)
                
                # キーポイント名を表示するオプション（必要に応じてコメントを外す）
                cv2.putText(
                   result_img,
                   KEYPOINT_NAMES[kpt_idx],
                   (kx_scaled + 5, ky_scaled),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.4,
                   color,
                   1,
                   cv2.LINE_AA
                )
    
    return result_img, valid_detections

def main():
    parser = argparse.ArgumentParser(description='YOLOv8-poseモデルのONNXを使用してポーズ検出を行います')
    parser.add_argument('--model', type=str, required=True, help='ONNXモデルへのパス')
    parser.add_argument('--image', type=str, required=True, help='入力画像へのパス')
    parser.add_argument('--size', type=int, default=640, help='入力サイズ (デフォルト: 640)')
    parser.add_argument('--conf', type=float, default=0.25, help='検出信頼度閾値 (デフォルト: 0.25)')
    parser.add_argument('--kpt-conf', type=float, default=0.2, help='キーポイント信頼度閾値 (デフォルト: 0.2)')
    parser.add_argument('--output', type=str, default='pose_output.jpg', help='出力画像のパス (デフォルト: pose_output.jpg)')
    parser.add_argument('--show', action='store_true', help='結果をウィンドウに表示する')
    parser.add_argument('--save-numpy', type=str, default='', help='outputs データをNumPy形式で保存するパス')
    
    args = parser.parse_args()
    
    # モデルをロード
    session = load_model(args.model)
    if session is None:
        return
    
    # 画像を準備
    input_data, original_size = prepare_image(args.image, args.size)
    if input_data is None:
        return
    
    # 推論を実行
    outputs = run_inference(session, input_data)
    if outputs is None:
        return
        
    # NumPy配列として推論結果を保存（オプション）
    if args.save_numpy:
        try:
            np.save(args.save_numpy, outputs)
            print(f"推論結果を保存しました: {args.save_numpy}")
            print(f"出力形状: {[output.shape for output in outputs]}")
        except Exception as e:
            print(f"NumPy形式での保存中にエラーが発生しました: {e}")
    
    # ポーズ検出結果を処理
    result_img, detections = process_pose_detections(
        outputs, original_size, args.size, args.conf, args.kpt_conf
    )
    
    # 結果を表示
    print(f"検出された人物数: {len(detections)}")
    
    # 結果を保存
    cv2.imwrite(args.output, result_img)
    print(f"結果を保存しました: {args.output}")
    
    # 結果を表示（オプション）
    if args.show:
        cv2.imshow('Pose Detection Result', result_img)
        print("キーを押して終了します...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"結果は {args.output} に保存されました。--show オプションを使用すると結果が表示されます。")

if __name__ == "__main__":
    main()
