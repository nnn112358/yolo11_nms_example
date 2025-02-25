import onnxruntime as ort
import cv2
import numpy as np
import argparse
import os

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

def process_detections(outputs, original_size, input_size=640, conf_threshold=0.25, format='xyxy'):
    """
    検出結果を処理して表示する関数
    format: 座標形式 ('xywh': 中心座標+幅高さ, 'xyxy': 左上+右下座標)
    """
    original_width, original_height, original_img = original_size
    
    # 出力が複数ある場合があるため、最初の出力を使用
    detections = outputs[0]
    print(f"検出形状: {detections.shape}")
    
    # 出力形式に合わせて処理
    if len(detections.shape) == 4:  # [1, 1, boxes, attrs]
        detections = detections[0, 0]
    elif len(detections.shape) == 3:  # [1, boxes, attrs]
        detections = detections[0]
    
    # 閾値を超える検出結果のみを保持
    valid_detections = []
    for detection in detections:
        if len(detection) >= 6:  # 座標とスコアとクラス
            score = detection[4]
            if score > conf_threshold:
                valid_detections.append(detection)
    
    result_img = original_img.copy()
    
    # クラスごとに異なる色を生成
    class_colors = {}
    np.random.seed(42)  # 色の再現性のため
    
    # 入力サイズから元の画像サイズへのスケーリング係数
    scale_x = original_width / input_size
    scale_y = original_height / input_size
    
    # 検出結果を描画
    for detection in valid_detections:
        # 座標情報とスコア、クラス
        coords = detection[:4]
        score = detection[4]
        class_id = int(detection[5])
        
        # クラスごとに色を割り当て
        if class_id not in class_colors:
            class_colors[class_id] = (
                np.random.randint(0, 255),
                np.random.randint(0, 255),
                np.random.randint(0, 255)
            )
        color = class_colors[class_id]
        
        # 座標系に応じた座標変換
        if format == 'xywh':
            # xywh形式（中心x, 中心y, 幅, 高さ）
            x_center, y_center, width, height = coords
            
            # 入力サイズから元の画像サイズにスケーリング
            x_center = x_center * scale_x
            y_center = y_center * scale_y
            width = width * scale_x
            height = height * scale_y
            
            # 中心座標から左上と右下の座標を計算
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)
            
        elif format == 'xyxy':
            # xyxy形式（左上x, 左上y, 右下x, 右下y）
            x1, y1, x2, y2 = coords
            
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
        
        # 境界ボックスを描画
        cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
        
        # スコアとクラスIDを表示するラベルの背景を描画
        label = f"Class: {class_id}, {score:.2f}"
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        
        # ラベルの位置調整（画像上部の境界を超えないように）
        y1_label = max(0, y1 - label_height - 10)
        
        cv2.rectangle(
            result_img,
            (x1, y1_label),
            (x1 + label_width, y1),
            color,
            -1,  # 塗りつぶし
        )
        
        # ラベル（クラスIDとスコア）を表示
        cv2.putText(
            result_img,
            label,
            (x1, y1 - 7 if y1 > label_height + 10 else y1 + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),  # 白色テキスト
            1,
            cv2.LINE_AA  # アンチエイリアス
        )
    
    return result_img, valid_detections

def main():
    parser = argparse.ArgumentParser(description='YOLOモデルのONNXを使用して物体検出を行います')
    parser.add_argument('--model', type=str, required=True, help='ONNXモデルへのパス')
    parser.add_argument('--image', type=str, required=True, help='入力画像へのパス')
    parser.add_argument('--size', type=int, default=640, help='入力サイズ (デフォルト: 640)')
    parser.add_argument('--conf', type=float, default=0.25, help='信頼度閾値 (デフォルト: 0.25)')
    parser.add_argument('--output', type=str, default='output.jpg', help='出力画像のパス (デフォルト: output.jpg)')
    parser.add_argument('--format', type=str, default='xyxy', choices=['xywh', 'xyxy'], 
                        help='出力座標形式 (xywh: 中心+幅高さ, xyxy: 左上+右下)')
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
    
    # 検出結果を処理
    result_img, detections = process_detections(
        outputs, original_size, args.size, args.conf, args.format
    )
    
    # 結果を表示
    print(f"検出された物体数: {len(detections)}")
    for i, detection in enumerate(detections):
        coords = detection[:4]
        score = detection[4]
        class_id = int(detection[5])
        
        if args.format == 'xywh':
            x_center, y_center, width, height = coords
            print(f"検出 {i+1}: クラス={class_id}, スコア={score:.4f}, "
                  f"中心=({x_center:.1f}, {y_center:.1f}), サイズ={width:.1f}x{height:.1f}")
        else:  # xyxy
            x1, y1, x2, y2 = coords
            print(f"検出 {i+1}: クラス={class_id}, スコア={score:.4f}, "
                  f"座標=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
    
    # 結果を保存
    cv2.imwrite(args.output, result_img)
    print(f"結果を保存しました: {args.output}")
    
    # 結果を表示（オプション）
    if args.show:
        cv2.imshow('Detection Result', result_img)
        print("キーを押して終了します...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"結果は {args.output} に保存されました。--show オプションを使用すると結果が表示されます。")

if __name__ == "__main__":
    main()