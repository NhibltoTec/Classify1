from roboflow import Roboflow

print("🔑 Khởi tạo Roboflow...")
rf = Roboflow(api_key="NlO0Js6OIvklw9RXfvRN")
project = rf.workspace("aitest1-ry9lo").project("classify-waste")
version = project.version(1)

print("⬇️ Bắt đầu tải dataset...")
dataset = version.download("folder")
print("✅ Dataset tải xong:", dataset.location)
