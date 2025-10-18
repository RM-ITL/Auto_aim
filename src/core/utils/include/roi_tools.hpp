#ifndef UTILS_ROI_MANAGER_HPP
#define UTILS_ROI_MANAGER_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>

namespace utils {

/**
 * ROI处理的元数据
 */
struct ROIMetadata {
    // ROI在原图中的位置
    int offset_x = 0;
    int offset_y = 0;
    
    // ROI的实际尺寸
    int roi_width = 0;      
    int roi_height = 0;
    
    // 输出尺寸
    int output_width = 0;   
    int output_height = 0;
    
    // 缩放系数：output_size / roi_size
    double scale_x = 1.0;   
    double scale_y = 1.0;
    
    // 原图尺寸
    int original_width = 0;
    int original_height = 0;
    
    double getAverageScale() const {
        return (scale_x + scale_y) / 2.0;
    }
    
    bool isValid() const {
        return roi_width > 0 && roi_height > 0 && 
               output_width > 0 && output_height > 0;
    }
};

/**
 * ROI管理器 - 处理图像ROI提取和坐标转换
 */
class ROIManager {
public:
    ROIManager(int target_size = -1, int roi_size = 600) 
        : target_size_(target_size), roi_size_(roi_size) {
        if (roi_size_ <= 0) {
            roi_size_ = 600;
        }
    }
    
    /**
     * 从图像中心提取ROI
     */
    cv::Mat extractCenterROI(const cv::Mat& image, ROIMetadata& metadata) {
        if (image.empty()) {
            metadata = ROIMetadata();
            return cv::Mat();
        }
        
        metadata.original_width = image.cols;
        metadata.original_height = image.rows;
        
        int center_x = image.cols / 2;
        int center_y = image.rows / 2;
        
        int roi_x = center_x - roi_size_ / 2;
        int roi_y = center_y - roi_size_ / 2;
        
        roi_x = std::max(0, roi_x);
        roi_y = std::max(0, roi_y);
        
        int actual_width = std::min(roi_size_, image.cols - roi_x);
        int actual_height = std::min(roi_size_, image.rows - roi_y);
        
        metadata.offset_x = roi_x;
        metadata.offset_y = roi_y;
        metadata.roi_width = actual_width;
        metadata.roi_height = actual_height;
        
        cv::Rect roi_rect(roi_x, roi_y, actual_width, actual_height);
        cv::Mat roi = image(roi_rect).clone();
        
        // 边界填充
        if (actual_width < roi_size_ || actual_height < roi_size_) {
            cv::Mat padded_roi = cv::Mat::zeros(roi_size_, roi_size_, image.type());
            roi.copyTo(padded_roi(cv::Rect(0, 0, actual_width, actual_height)));
            roi = padded_roi;
            metadata.roi_width = roi_size_;
            metadata.roi_height = roi_size_;
        }
        
        // resize到目标尺寸
        if (target_size_ > 0 && target_size_ != roi_size_) {
            cv::resize(roi, roi, cv::Size(target_size_, target_size_));
            metadata.output_width = target_size_;
            metadata.output_height = target_size_;
            metadata.scale_x = static_cast<double>(target_size_) / metadata.roi_width;
            metadata.scale_y = static_cast<double>(target_size_) / metadata.roi_height;
        } else {
            metadata.output_width = metadata.roi_width;
            metadata.output_height = metadata.roi_height;
            metadata.scale_x = 1.0;
            metadata.scale_y = 1.0;
        }
        
        return roi;
    }
    
    /**
     * 从指定位置提取ROI
     */
    cv::Mat extractROI(const cv::Mat& image, int x, int y, ROIMetadata& metadata) {
        if (image.empty()) {
            metadata = ROIMetadata();
            return cv::Mat();
        }
        
        metadata.original_width = image.cols;
        metadata.original_height = image.rows;
        
        x = std::max(0, x);
        y = std::max(0, y);
        
        int actual_width = std::min(roi_size_, image.cols - x);
        int actual_height = std::min(roi_size_, image.rows - y);
        
        metadata.offset_x = x;
        metadata.offset_y = y;
        metadata.roi_width = actual_width;
        metadata.roi_height = actual_height;
        
        cv::Rect roi_rect(x, y, actual_width, actual_height);
        cv::Mat roi = image(roi_rect).clone();
        
        if (actual_width < roi_size_ || actual_height < roi_size_) {
            cv::Mat padded_roi = cv::Mat::zeros(roi_size_, roi_size_, image.type());
            roi.copyTo(padded_roi(cv::Rect(0, 0, actual_width, actual_height)));
            roi = padded_roi;
            metadata.roi_width = roi_size_;
            metadata.roi_height = roi_size_;
        }
        
        if (target_size_ > 0 && target_size_ != roi_size_) {
            cv::resize(roi, roi, cv::Size(target_size_, target_size_));
            metadata.output_width = target_size_;
            metadata.output_height = target_size_;
            metadata.scale_x = static_cast<double>(target_size_) / metadata.roi_width;
            metadata.scale_y = static_cast<double>(target_size_) / metadata.roi_height;
        } else {
            metadata.output_width = metadata.roi_width;
            metadata.output_height = metadata.roi_height;
            metadata.scale_x = 1.0;
            metadata.scale_y = 1.0;
        }
        
        return roi;
    }
    
    /**
     * ROI坐标转原图坐标
     */
    cv::Point2f transformToOriginal(const cv::Point2f& roi_point, 
                                    const ROIMetadata& metadata) const {
        if (!metadata.isValid()) {
            return roi_point;
        }
        
        cv::Point2f point_in_roi;
        point_in_roi.x = roi_point.x / metadata.scale_x;
        point_in_roi.y = roi_point.y / metadata.scale_y;
        
        cv::Point2f original_point;
        original_point.x = point_in_roi.x + metadata.offset_x;
        original_point.y = point_in_roi.y + metadata.offset_y;
        
        return original_point;
    }
    
    std::vector<cv::Point2f> transformToOriginal(const std::vector<cv::Point2f>& roi_points,
                                                 const ROIMetadata& metadata) const {
        std::vector<cv::Point2f> original_points;
        original_points.reserve(roi_points.size());
        
        for (const auto& point : roi_points) {
            original_points.push_back(transformToOriginal(point, metadata));
        }
        
        return original_points;
    }
    
    /**
     * 原图坐标转ROI坐标
     */
    cv::Point2f transformToROI(const cv::Point2f& original_point,
                               const ROIMetadata& metadata) const {
        if (!metadata.isValid()) {
            return original_point;
        }
        
        cv::Point2f point_in_roi;
        point_in_roi.x = original_point.x - metadata.offset_x;
        point_in_roi.y = original_point.y - metadata.offset_y;
        
        cv::Point2f roi_point;
        roi_point.x = point_in_roi.x * metadata.scale_x;
        roi_point.y = point_in_roi.y * metadata.scale_y;
        
        return roi_point;
    }
    
    bool isPointInROI(const cv::Point2f& original_point,
                      const ROIMetadata& metadata) const {
        if (!metadata.isValid()) {
            return false;
        }
        
        return original_point.x >= metadata.offset_x &&
               original_point.x < metadata.offset_x + metadata.roi_width &&
               original_point.y >= metadata.offset_y &&
               original_point.y < metadata.offset_y + metadata.roi_height;
    }
    
    void drawROIBoundary(cv::Mat& image, const ROIMetadata& metadata,
                        const cv::Scalar& color = cv::Scalar(0, 255, 255),
                        int thickness = 2) const {
        if (!metadata.isValid() || image.empty()) {
            return;
        }
        
        cv::Rect roi_rect(metadata.offset_x, metadata.offset_y,
                          metadata.roi_width, metadata.roi_height);
        cv::rectangle(image, roi_rect, color, thickness);
        
        std::string scale_info = cv::format("Scale: %.2fx", metadata.getAverageScale());
        cv::putText(image, scale_info, 
                   cv::Point(metadata.offset_x + 5, metadata.offset_y - 5),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
    }
    
    int getROISize() const { return roi_size_; }
    int getTargetSize() const { return target_size_; }
    
    void setROISize(int size) { 
        if (size > 0) {
            roi_size_ = size;
        }
    }
    
    void setTargetSize(int size) { 
        target_size_ = size;
    }
    
private:
    int target_size_;  // 目标输出尺寸（-1表示不resize）
    int roi_size_;     // ROI尺寸
};

} // namespace utils

#endif // UTILS_ROI_MANAGER_HPP